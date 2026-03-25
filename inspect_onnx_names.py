import onnx
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Шлях до вашої ONNX моделі
ONNX_MODEL_PATH = "runs\\rfdetr_medium\\baseline\\weights\\inference_model.sim.onnx"

# Для конвертації в TensorRT (Jetson Orin) можна використовувати цю команду:
"""
trtexec command example:
/usr/src/tensorrt/bin/trtexec \
  --onnx=inference_model.onnx \
  --saveEngine=model.trt \
  --fp16 \
  --precisionConstraints=obey \
  --layerPrecisions=\
/downsample_layers.0/downsample_layers.0.1/ReduceMean_1:fp32,\
/downsample_layers.0/downsample_layers.0.1/ReduceMean:fp32,\
/downsample_layers.0/downsample_layers.0.1/Pow:fp32,\
/downsample_layers.1/downsample_layers.1.0/ReduceMean:fp32,\
/downsample_layers.1/downsample_layers.1.0/Pow:fp32,\
/downsample_layers.1/downsample_layers.1.0/ReduceMean_1:fp32,\
/downsample_layers.2/downsample_layers.2.0/ReduceMean:fp32,\
/downsample_layers.2/downsample_layers.2.0/Pow:fp32,\
/downsample_layers.2/downsample_layers.2.0/ReduceMean_1:fp32,\
/downsample_layers.3/downsample_layers.3.0/ReduceMean:fp32,\
/downsample_layers.3/downsample_layers.3.0/Pow:fp32,\
/downsample_layers.3/downsample_layers.3.0/ReduceMean_1:fp32
"""



def inspect_onnx_names(model_path):
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found.")
        return

    print(f"Loading ONNX model: {model_path}...")
    onnx_model = onnx.load(model_path)
    
    # Визначаємо шлях для вихідного файлу
    base_dir = os.path.dirname(os.path.abspath(model_path))
    model_name = os.path.basename(model_path)
    output_txt = os.path.join(base_dir, f"{model_name}_layer_names.txt")
    
    node_names = []
    problematic_nodes = [] # ReduceMean, Pow, Softmax

    for node in onnx_model.graph.node:
        name = node.name if node.name else f"unnamed_{node.op_type}"
        node_names.append(name)
        
        if node.op_type in ["ReduceMean", "Pow", "Softmax"]:
            problematic_nodes.append(f"{name} ({node.op_type})")

    # Зберігаємо всі імена у файл
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write("-" * 50 + "\n")
        f.write("POTENTIALLY PROBLEMATIC LAYERS (Recommend FP32 for TRT):\n")
        for p_node in problematic_nodes:
            f.write(f"  - {p_node}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("ALL LAYER NAMES:\n")
        for name in node_names:
            f.write(f"{name}\n")

    print(f"Found {len(node_names)} layers.")
    print(f"Found {len(problematic_nodes)} problematic layers (ReduceMean/Pow/Softmax).")
    print(f"Results saved to: {output_txt}")

if __name__ == "__main__":
    inspect_onnx_names(ONNX_MODEL_PATH)

