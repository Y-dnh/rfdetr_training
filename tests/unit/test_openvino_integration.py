from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv
import torch

from rfdetr.detr import RFDETR
from rfdetr.deploy.openvino import (
    OpenVINOExportMetadata,
    OpenVINOInferenceModel,
    load_openvino_metadata,
    save_openvino_metadata,
)
from rfdetr.training import ExportConfig, RFDETRTrainer


def test_export_config_accepts_openvino_roundtrip():
    config = ExportConfig(format="openvino", ov_compress_to_fp16=False)

    payload = config.to_dict()

    assert payload["format"] == "openvino"
    assert payload["ov_compress_to_fp16"] is False
    assert ExportConfig.from_dict(payload) == config


def test_openvino_metadata_roundtrip(tmp_path: Path):
    metadata = OpenVINOExportMetadata(
        backend="openvino",
        resolution=576,
        batch_size=1,
        class_names=["car"],
        num_select=10,
        input_name="input",
        output_names=["dets", "labels"],
        source_checkpoint="weights/best.pt",
        compress_to_fp16=True,
    )

    metadata_path = tmp_path / "inference_model.metadata.json"
    save_openvino_metadata(metadata, metadata_path)
    restored = load_openvino_metadata(metadata_path)

    assert restored == metadata


def test_openvino_wrapper_maps_outputs_to_tuple():
    class FakeCompiledModel:
        def __init__(self):
            self._ports = {
                "dets": object(),
                "labels": object(),
            }
            self.last_inputs = None

        def __call__(self, inputs):
            self.last_inputs = inputs
            return {
                self._ports["labels"]: np.array([[[7.0]]], dtype=np.float32),
                self._ports["dets"]: np.array([[[0.5, 0.5, 0.25, 0.25]]], dtype=np.float32),
            }

        def output(self, key):
            if isinstance(key, str):
                return self._ports[key]
            return [self._ports["dets"], self._ports["labels"]][key]

    model = OpenVINOInferenceModel.__new__(OpenVINOInferenceModel)
    model.batch_size = 1
    model.input_name = "input"
    model.output_names = ["dets", "labels"]
    model.compiled_model = FakeCompiledModel()

    outputs = model(torch.zeros((1, 3, 576, 576), dtype=torch.float32))

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert tuple(outputs[0].shape) == (1, 1, 4)
    assert tuple(outputs[1].shape) == (1, 1, 1)
    assert "input" in model.compiled_model.last_inputs


def test_from_openvino_predict_returns_detections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata = OpenVINOExportMetadata(
        backend="openvino",
        resolution=576,
        batch_size=1,
        class_names=["car"],
        num_select=1,
        input_name="input",
        output_names=["dets", "labels"],
        source_checkpoint=None,
        compress_to_fp16=True,
    )

    model_path = tmp_path / "inference_model.xml"
    model_path.write_text("<xml/>", encoding="utf-8")
    metadata_path = tmp_path / "inference_model.metadata.json"
    save_openvino_metadata(metadata, metadata_path)

    class FakeOpenVINOInferenceModel:
        def __init__(self, model_path, device, metadata):
            self.model_path = model_path
            self.device = device
            self.metadata = metadata

        def __call__(self, input_tensor: torch.Tensor):
            batch_size = input_tensor.shape[0]
            boxes = torch.tensor(
                [[[0.5, 0.5, 0.25, 0.25]]],
                dtype=torch.float32,
            ).repeat(batch_size, 1, 1)
            logits = torch.tensor(
                [[[12.0]]],
                dtype=torch.float32,
            ).repeat(batch_size, 1, 1)
            return boxes, logits

    monkeypatch.setattr("rfdetr.deploy.openvino.OpenVINOInferenceModel", FakeOpenVINOInferenceModel)

    model = RFDETR.from_openvino(model_path, metadata_path=metadata_path)
    detections = model.predict(torch.zeros((3, 480, 640), dtype=torch.float32), threshold=0.5)

    assert isinstance(detections, sv.Detections)
    assert len(detections) == 1


def test_from_openvino_enforces_static_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata = OpenVINOExportMetadata(
        backend="openvino",
        resolution=576,
        batch_size=1,
        class_names=["car"],
        num_select=1,
        input_name="input",
        output_names=["dets", "labels"],
        source_checkpoint=None,
        compress_to_fp16=True,
    )

    model_path = tmp_path / "inference_model.xml"
    model_path.write_text("<xml/>", encoding="utf-8")
    metadata_path = tmp_path / "inference_model.metadata.json"
    save_openvino_metadata(metadata, metadata_path)

    class FakeOpenVINOInferenceModel:
        def __init__(self, model_path, device, metadata):
            self.model_path = model_path
            self.device = device
            self.metadata = metadata

        def __call__(self, input_tensor: torch.Tensor):
            raise AssertionError("Inference wrapper should not be called on batch mismatch.")

    monkeypatch.setattr("rfdetr.deploy.openvino.OpenVINOInferenceModel", FakeOpenVINOInferenceModel)

    model = RFDETR.from_openvino(model_path, metadata_path=metadata_path)

    with pytest.raises(ValueError, match="Batch size mismatch"):
        model.predict(
            [
                torch.zeros((3, 480, 640), dtype=torch.float32),
                torch.zeros((3, 480, 640), dtype=torch.float32),
            ]
        )


def test_trainer_export_dispatcher_reports_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trainer = RFDETRTrainer(export_config=ExportConfig(format="openvino"))
    trainer.save_dir = tmp_path
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.pt").write_text("stub", encoding="utf-8")
    trainer.training_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    trainer.model = object()
    trainer.class_names = ["car"]
    trainer.postprocessor = SimpleNamespace(num_select=25)

    expected = {
        "onnx_path": str(tmp_path / "weights" / "inference_model.sim.onnx"),
        "openvino_path": str(tmp_path / "weights" / "inference_model.xml"),
        "openvino_bin_path": str(tmp_path / "weights" / "inference_model.bin"),
        "openvino_metadata_path": str(tmp_path / "weights" / "inference_model.metadata.json"),
    }

    def fake_export_artifacts_from_checkpoint(**kwargs):
        assert kwargs["export_format"] == "openvino"
        assert kwargs["class_names"] == ["car"]
        assert kwargs["num_select"] == 25
        return expected

    monkeypatch.setattr(
        "rfdetr.deploy.export.export_artifacts_from_checkpoint",
        fake_export_artifacts_from_checkpoint,
    )

    artifacts = trainer._export_model_artifacts()

    assert artifacts == expected
