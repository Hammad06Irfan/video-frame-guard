import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_DIR = "models"
INPUT_MODEL = os.path.join(MODEL_DIR, "model.onnx")
OUTPUT_MODEL = os.path.join(MODEL_DIR, "model_int8.onnx")

def run_quantization():
    if not os.path.exists(INPUT_MODEL):
        print(f"[-] Error: {INPUT_MODEL} not found. Run src/train.py first.")
        return

    print(f"[*] Starting dynamic INT8 quantization on {INPUT_MODEL}...")

    # Load and clean graph value info to avoid dimension collision during shape inference
    model = onnx.load(INPUT_MODEL)
    
    # Strip existing intermediate shape hints so the quantizer infers clean INT8 shapes
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
    
    clean_model_path = os.path.join(MODEL_DIR, "model_clean.onnx")
    onnx.save(model, clean_model_path)

    # Quantize FP32 weights to INT8 dynamically
    quantize_dynamic(
        model_input=clean_model_path,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QInt8,
        extra_options={"DisableShapeInference": True}
    )

    # Clean up intermediate file
    if os.path.exists(clean_model_path):
        os.remove(clean_model_path)

    # Compare file sizes
    fp32_size = os.path.getsize(INPUT_MODEL) / (1024 * 1024)
    int8_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    reduction = ((fp32_size - int8_size) / fp32_size) * 100

    print(f"[+] Quantization complete!")
    print(f"    - Original FP32 Model: {fp32_size:.2f} MB")
    print(f"    - Quantized INT8 Model: {int8_size:.2f} MB")
    print(f"    - Size Reduction:      {reduction:.1f}%")

if __name__ == "__main__":
    run_quantization()