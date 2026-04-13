from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch


def _import_openvino():
    try:
        import openvino as ov
    except ImportError as exc:
        raise ImportError(
            "OpenVINO support requires the `openvino` package. "
            "Install it with `pip install openvino`."
        ) from exc
    return ov


PathLike = Union[str, Path]


@dataclass
class OpenVINOExportMetadata:
    """Serializable metadata required to run RF-DETR inference from OpenVINO IR."""

    backend: str
    resolution: int
    batch_size: int
    class_names: list[str]
    num_select: int
    input_name: str
    output_names: list[str]
    source_checkpoint: Optional[str]
    compress_to_fp16: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "OpenVINOExportMetadata":
        """Build metadata from a dictionary."""
        return cls(
            backend=str(value["backend"]),
            resolution=int(value["resolution"]),
            batch_size=int(value["batch_size"]),
            class_names=list(value.get("class_names", [])),
            num_select=int(value["num_select"]),
            input_name=str(value["input_name"]),
            output_names=list(value["output_names"]),
            source_checkpoint=value.get("source_checkpoint"),
            compress_to_fp16=bool(value.get("compress_to_fp16", True)),
        )


def resolve_openvino_model_path(model_path: PathLike) -> Path:
    """Resolve an OpenVINO IR path from either a file path or an export directory."""
    resolved = Path(model_path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "inference_model.xml"
    if resolved.suffix.lower() != ".xml":
        raise ValueError(f"Expected an OpenVINO XML path, got: {resolved}")
    return resolved


def infer_metadata_path(model_path: PathLike) -> Path:
    """Return the default metadata path located next to the OpenVINO IR XML file."""
    return resolve_openvino_model_path(model_path).with_suffix(".metadata.json")


def save_openvino_metadata(metadata: OpenVINOExportMetadata, metadata_path: PathLike) -> str:
    """Persist OpenVINO export metadata as JSON."""
    resolved = Path(metadata_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2)
    return str(resolved)


def load_openvino_metadata(metadata_path: PathLike) -> OpenVINOExportMetadata:
    """Load OpenVINO export metadata from JSON."""
    resolved = Path(metadata_path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return OpenVINOExportMetadata.from_dict(payload)


def export_openvino_ir(
    onnx_path: PathLike,
    output_dir: PathLike,
    metadata: OpenVINOExportMetadata,
    compress_to_fp16: bool = True,
) -> Dict[str, str]:
    """Convert ONNX to OpenVINO IR and save the companion metadata sidecar."""
    ov = _import_openvino()

    onnx_resolved = Path(onnx_path).expanduser().resolve()
    output_resolved = Path(output_dir).expanduser().resolve()
    output_resolved.mkdir(parents=True, exist_ok=True)

    model = ov.convert_model(str(onnx_resolved))
    xml_path = output_resolved / "inference_model.xml"
    ov.save_model(model, str(xml_path), compress_to_fp16=compress_to_fp16)

    metadata_path = xml_path.with_suffix(".metadata.json")
    save_openvino_metadata(metadata, metadata_path)

    return {
        "xml_path": str(xml_path),
        "bin_path": str(xml_path.with_suffix(".bin")),
        "metadata_path": str(metadata_path),
    }


class OpenVINOInferenceModel:
    """Callable wrapper around a compiled OpenVINO model returning RF-DETR tuple outputs."""

    def __init__(
        self,
        model_path: PathLike,
        device: str = "CPU",
        metadata: Optional[OpenVINOExportMetadata] = None,
    ) -> None:
        ov = _import_openvino()

        self.model_path = resolve_openvino_model_path(model_path)
        self.metadata = metadata or load_openvino_metadata(infer_metadata_path(self.model_path))
        self.device = device
        self.batch_size = self.metadata.batch_size
        self.input_name = self.metadata.input_name
        self.output_names = list(self.metadata.output_names)

        self.core = ov.Core()
        self.model = self.core.read_model(str(self.model_path))
        self.compiled_model = self.core.compile_model(self.model, device)

    def __call__(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run OpenVINO inference and return CPU tensors matching the exported RF-DETR tuple."""
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("OpenVINO inference expects a torch.Tensor input.")
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected a 4D input tensor, got shape {tuple(input_tensor.shape)}.")
        if input_tensor.shape[0] != self.batch_size:
            raise ValueError(
                f"Batch size mismatch. OpenVINO model expects batch size {self.batch_size}, "
                f"but got {input_tensor.shape[0]}."
            )

        input_array = input_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        result = self.compiled_model({self.input_name: input_array})

        outputs = []
        for index, name in enumerate(self.output_names):
            try:
                output_port = self.compiled_model.output(name)
            except Exception:
                output_port = self.compiled_model.output(index)
            outputs.append(torch.from_numpy(np.asarray(result[output_port])))
        return tuple(outputs)
