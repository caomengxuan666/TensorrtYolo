#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <execution>
#include <future>
#include <algorithm>
using namespace nvinfer1;

// Logger implementation
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

struct Detection {
    int class_id;
    float score;
    cv::Rect box;

    // For sorting
    bool operator<(const Detection& other) const {
        return score > other.score; // Descending order
    }
};

// Standalone NMS function
std::vector<Detection> nonMaximumSuppression(std::vector<Detection>& detections, float threshold) {
    if (detections.empty()) return {};

    // Sort by score in descending order
    std::sort(std::execution::par, detections.begin(), detections.end());

    std::vector<float> areas;
    areas.reserve(detections.size());
    for (const auto& det : detections) {
        areas.push_back(det.box.width * det.box.height);
    }

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> results;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        results.push_back(detections[i]);

        // Process in parallel where possible
        std::for_each(std::execution::par, detections.begin() + i + 1, detections.end(),
            [&](const Detection& det_j) {
                size_t j = &det_j - detections.data();
                if (suppressed[j]) return;

                const auto& a = detections[i].box;
                const auto& b = det_j.box;

                float xx1 = std::max(a.x, b.x);
                float yy1 = std::max(a.y, b.y);
                float xx2 = std::min(a.x + a.width, b.x + b.width);
                float yy2 = std::min(a.y + a.height, b.y + b.height);

                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float inter = w * h;

                float ovr = inter / (areas[i] + areas[j] - inter);

                if (ovr > threshold) suppressed[j] = true;
            });
    }
    return results;
}

// Asynchronous file loading
std::future<std::vector<char>> loadEngineFileAsync(const std::string& engineFile) {
    return std::async(std::launch::async, [=] {
        std::ifstream file(engineFile, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open engine file");
        return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        });
}

// Parallel preprocessing
#include <execution>  // C++17 并行算法

void preprocessParallel(const cv::Mat& img, float* dst, int input_w, int input_h, float& scale, int& dw, int& dh) {
    int w = img.cols, h = img.rows;
    scale = std::min(input_w / (float)w, input_h / (float)h);
    int new_w = int(w * scale), new_h = int(h * scale);
    dw = (input_w - new_w) / 2;
    dh = (input_h - new_h) / 2;

    cv::Mat resized, padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(img, resized, cv::Size(new_w, new_h));
    resized.copyTo(padded(cv::Rect(dw, dh, new_w, new_h)));

    // 使用 C++17 并行算法处理每个通道
    for (int c = 0; c < 3; ++c) {
        std::for_each(std::execution::par, padded.begin<cv::Vec3b>(), padded.end<cv::Vec3b>(),
            [=, &padded](cv::Vec3b& pixel) {
                int y = &pixel - padded.ptr<cv::Vec3b>(0);  // 计算当前行
                int x = (y % input_h);                     // 计算当前列
                y /= input_h;
                dst[c * input_h * input_w + y * input_w + x] = pixel[c] / 255.0f;
            });
    }
}

std::string roundDecimalToString(float value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
    return oss.str();
}

int main() {
    const std::string engineFile = "best.engine";
    const std::string imageFile = "test_image.png";
    constexpr int INPUT_W = 640, INPUT_H = 640;
    const float CONF_THRESH = 0.25f, NMS_THRESH = 0.45f;
    const int NUM_BOXES = 8400;

    // 1. Asynchronously load engine file
    auto engineFuture = loadEngineFileAsync(engineFile);

    // Load image while engine is loading
    cv::Mat img = cv::imread(imageFile);
    if (img.empty()) {
        std::cerr << "Image not found!\n";
        return -1;
    }

    // Get engine data (this will wait if not ready)
    auto engineData = engineFuture.get();

    // 2. Create runtime and engine
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

    // 3. Get input/output names
    std::string inputName, outputName;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inputName = name;
        }
        else {
            outputName = name;
        }
    }

    // 4. Allocate device memory
    size_t inputSize = 1 * 3 * INPUT_H * INPUT_W * sizeof(float);
    size_t outputSize = 1 * 5 * NUM_BOXES * sizeof(float);

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // 5. Preprocess (parallel)
    float scale; int dw, dh;
    std::vector<float> inputData(3 * INPUT_H * INPUT_W);
    preprocessParallel(img, inputData.data(), INPUT_W, INPUT_H, scale, dw, dh);

    // 6. Async memory copy
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream);

    // 7. Set tensor addresses
    context->setInputTensorAddress(inputName.c_str(), d_input);
    context->setOutputTensorAddress(outputName.c_str(), d_output);

    // 8. Async inference
    context->enqueueV3(stream);

    // 9. Prepare output buffer and async copy
    std::vector<float> outputData(5 * NUM_BOXES);
    cudaMemcpyAsync(outputData.data(), d_output, outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Debug output
    std::cout << "Input tensor first 10 values: ";
    for (int i = 0; i < 10; ++i) std::cout << inputData[i] << " ";
    std::cout << "\nOutput tensor first 10 values: ";
    for (int i = 0; i < 10; ++i) std::cout << outputData[i] << " ";
    std::cout << std::endl;

    // 10. Transpose and filter detections
    std::vector<Detection> detections;
    detections.reserve(NUM_BOXES);

    // Parallel processing of detections
    std::vector<std::future<void>> detection_futures;
    const int num_threads = std::thread::hardware_concurrency();
    const int chunk_size = NUM_BOXES / num_threads;
    std::mutex mtx;
    for (int t = 0; t < num_threads; ++t) {
        detection_futures.push_back(std::async(std::launch::async, [&, t] {
            const int start = t * chunk_size;
            const int end = (t == num_threads - 1) ? NUM_BOXES : (t + 1) * chunk_size;

            std::vector<Detection> local_dets;

            for (int i = start; i < end; ++i) {
                float cx = outputData[i];
                float cy = outputData[NUM_BOXES + i];
                float w = outputData[2 * NUM_BOXES + i];
                float h = outputData[3 * NUM_BOXES + i];
                float score = outputData[4 * NUM_BOXES + i];

                if (score < CONF_THRESH) continue;

                float x1 = (cx - w / 2 - dw) / scale;
                float y1 = (cy - h / 2 - dh) / scale;
                float x2 = (cx + w / 2 - dw) / scale;
                float y2 = (cy + h / 2 - dh) / scale;

                // Boundary checks
                x1 = std::max(0.f, std::min(x1, float(img.cols - 1)));
                y1 = std::max(0.f, std::min(y1, float(img.rows - 1)));
                x2 = std::max(0.f, std::min(x2, float(img.cols - 1)));
                y2 = std::max(0.f, std::min(y2, float(img.rows - 1)));

                local_dets.push_back({
                    0, // class_id
                    score,
                    cv::Rect(cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)))
                    });
            }

            // Merge results
            std::lock_guard<std::mutex> lock(mtx);
            detections.insert(detections.end(), local_dets.begin(), local_dets.end());
            }));
    }

    // Wait for all detection processing to complete
    for (auto& f : detection_futures) f.get();

    // 11. Apply NMS
    auto results = nonMaximumSuppression(detections, NMS_THRESH);

    // Debug output
    std::cout << "[DEBUG] Scores after NMS (first " << std::min(20, (int)results.size()) << "): ";
    for (int i = 0; i < std::min(20, (int)results.size()); ++i) {
        std::cout << results[i].score << " ";
    }
    std::cout << std::endl;

    // 12. Visualization
    for (const auto& det : results) {
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, "defect " + roundDecimalToString(det.score, 2),
            det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("TensorRT Inference", img);
    cv::waitKey(0);

    // 13. Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}