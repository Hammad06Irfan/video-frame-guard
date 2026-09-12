#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <onnxruntime_cxx_api.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// The 4 classes matching our training dataset alphabetical order:
const std::vector<std::string> LABELS = {"blocky", "clean", "noise", "torn"};

// Preprocessing: Loads image, normalizes pixels to ImageNet mean/std, converts HWC -> CHW
std::vector<float> preprocess_image(const char* image_path, int target_w, int target_h) {
    int w, h, channels;
    // Force load 3 channels (RGB)
    unsigned char* data = stbi_load(image_path, &w, &h, &channels, 3);
    if (!data) {
        std::cerr << "[-] Failed to load image: " << image_path << std::endl;
        exit(1);
    }

    // Standard ImageNet normalization parameters used in PyTorch
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    // Allocate memory for CHW tensor: [Channel, Height, Width]
    std::vector<float> tensor(3 * target_h * target_w);

    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            int src_pixel_idx = (y * target_w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                // Normalize 0..255 byte to 0.0..1.0 float, then apply (val - mean) / std
                float normalized = (data[src_pixel_idx + c] / 255.0f - mean[c]) / std_dev[c];
                // Store in Planar (CHW) order
                tensor[c * (target_h * target_w) + (y * target_w + x)] = normalized;
            }
        }
    }

    stbi_image_free(data);
    return tensor;
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/model.onnx";
    std::string test_image = (argc > 1) ? argv[1] : "data/dataset/val/torn/frame_0160_torn.png";

    std::cout << "==========================================\n";
    std::cout << " AMD Frame Guard - C++ Inference Engine   \n";
    std::cout << "==========================================\n";
    std::cout << "[*] Loading Model: " << model_path << "\n";
    std::cout << "[*] Target Frame:  " << test_image << "\n";

    // 1. Initialize ONNX Runtime Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AMDFrameGuard");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4); // Parallel CPU threads
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 2. Preprocess the frame into tensor memory
    std::vector<float> input_tensor_values = preprocess_image(test_image.c_str(), 224, 224);
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // 3. Warm-up run (clears instruction cache and sets up execution graphs)
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // 4. Benchmark Inference Latency over 50 iterations
    const int benchmark_runs = 50;
    std::vector<double> latencies;

    std::cout << "[*] Benchmarking inference over " << benchmark_runs << " iterations...\n";

    for (int i = 0; i < benchmark_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
        );

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        latencies.push_back(ms.count());
    }

    // 5. Compute Statistics
    double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / benchmark_runs;
    double fps = 1000.0 / avg_latency;

    // 6. Get Prediction
    auto final_output = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );
    float* scores = final_output[0].GetTensorMutableData<float>();

    // Softmax probabilities
    float exp_sum = 0.0f;
    for (int i = 0; i < 4; ++i) exp_sum += std::exp(scores[i]);

    int predicted_class = 0;
    float max_prob = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float prob = (std::exp(scores[i]) / exp_sum) * 100.0f;
        std::cout << "  - " << LABELS[i] << ": " << prob << "%\n";
        if (prob > max_prob) {
            max_prob = prob;
            predicted_class = i;
        }
    }

    std::cout << "\n------------------------------------------\n";
    std::cout << "[+] Result: Frame Classified as [" << LABELS[predicted_class] << "]\n";
    std::cout << "[+] Average Latency: " << avg_latency << " ms\n";
    std::cout << "[+] Throughput:      " << static_cast<int>(fps) << " FPS\n";
    std::cout << "------------------------------------------\n";

    return 0;
}