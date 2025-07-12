#include <benchmark/benchmark.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <future>
#include <algorithm>
#include <execution>
#include <mutex>
#include <thread>
#include <string>
#include <chrono>
#include <numeric>

using namespace nvinfer1;

// Logger implementation
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

// Detection struct
struct Detection {
    int class_id;
    float score;
    cv::Rect box;
    bool operator<(const Detection& other) const {
        return score > other.score;
    }
};

// NMS
std::vector<Detection> nonMaximumSuppression(std::vector<Detection>& detections, float threshold) {
    if (detections.empty()) return {};
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

// Engine file async load
std::future<std::vector<char>> loadEngineFileAsync(const std::string& engineFile) {
    return std::async(std::launch::async, [=] {
        std::ifstream file(engineFile, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open engine file");
        return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        });
}


// Preprocess
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

// Benchmark: 完整推理流程+详细阶段耗时
static void BM_TensorRT_FullPipeline(benchmark::State& state) {
    const std::string engineFile = "best.engine";
    const std::string imageFile = "test_image.png";
    constexpr int INPUT_W = 640, INPUT_H = 640;
    const float CONF_THRESH = 0.25f, NMS_THRESH = 0.45f;
    const int NUM_BOXES = 8400;

    // 只加载一次engine和图片
    static std::vector<char> engineData;
    static cv::Mat img;
    static bool loaded = false;
    static IRuntime* runtime = nullptr;
    static ICudaEngine* engine = nullptr;
    static IExecutionContext* context = nullptr;
    if (!loaded) {
        engineData = loadEngineFileAsync(engineFile).get();
        img = cv::imread(imageFile);
        if (img.empty()) throw std::runtime_error("Image not found!");
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        context = engine->createExecutionContext();
        loaded = true;
    }

    // 获取输入输出名
    std::string inputName, outputName;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) inputName = name;
        else outputName = name;
    }

    size_t inputSize = 1 * 3 * INPUT_H * INPUT_W * sizeof(float);
    size_t outputSize = 1 * 5 * NUM_BOXES * sizeof(float);

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // 累计各阶段耗时
    double total_pre = 0, total_h2d = 0, total_infer = 0, total_d2h = 0, total_post = 0, total_nms = 0;
    int rounds = 0;

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 预处理
        float scale; int dw, dh;
        std::vector<float> inputData(3 * INPUT_H * INPUT_W);
        auto t1 = std::chrono::high_resolution_clock::now();
        preprocessParallel(img, inputData.data(), INPUT_W, INPUT_H, scale, dw, dh);
        auto t2 = std::chrono::high_resolution_clock::now();

        // H2D
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice, stream);
        auto t3 = std::chrono::high_resolution_clock::now();

        // Set tensor addr
        context->setInputTensorAddress(inputName.c_str(), d_input);
        context->setOutputTensorAddress(outputName.c_str(), d_output);

        // 推理
        context->enqueueV3(stream);
        auto t4 = std::chrono::high_resolution_clock::now();

        // D2H
        std::vector<float> outputData(5 * NUM_BOXES);
        cudaMemcpyAsync(outputData.data(), d_output, outputSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        auto t5 = std::chrono::high_resolution_clock::now();

        // 后处理
        std::vector<Detection> detections;
        detections.reserve(NUM_BOXES);
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
                    x1 = std::max(0.f, std::min(x1, float(img.cols - 1)));
                    y1 = std::max(0.f, std::min(y1, float(img.rows - 1)));
                    x2 = std::max(0.f, std::min(x2, float(img.cols - 1)));
                    y2 = std::max(0.f, std::min(y2, float(img.rows - 1)));
                    local_dets.push_back({
                        0, score,
                        cv::Rect(cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)))
                        });
                }
                std::lock_guard<std::mutex> lock(mtx);
                detections.insert(detections.end(), local_dets.begin(), local_dets.end());
                }));
        }
        for (auto& f : detection_futures) f.get();
        auto t6 = std::chrono::high_resolution_clock::now();

        // NMS
        auto results = nonMaximumSuppression(detections, NMS_THRESH);
        auto t7 = std::chrono::high_resolution_clock::now();

        cudaStreamDestroy(stream);

        double pre = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double h2d = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double infer = std::chrono::duration<double, std::milli>(t4 - t3).count();
        double d2h = std::chrono::duration<double, std::milli>(t5 - t4).count();
        double post = std::chrono::duration<double, std::milli>(t6 - t5).count();
        double nms = std::chrono::duration<double, std::milli>(t7 - t6).count();

        total_pre += pre;
        total_h2d += h2d;
        total_infer += infer;
        total_d2h += d2h;
        total_post += post;
        total_nms += nms;
        rounds++;

        // 每轮打印
        std::cout << "[BENCH] preprocess: " << pre << " ms, H2D: " << h2d << " ms, infer: " << infer
            << " ms, D2H: " << d2h << " ms, post: " << post << " ms, NMS: " << nms << " ms" << std::endl;

        benchmark::ClobberMemory();
    }

    cudaFree(d_input);
    cudaFree(d_output);

    // 平均统计
    if (rounds > 0) {
        double sum = total_pre + total_h2d + total_infer + total_d2h + total_post + total_nms;
        std::cout << "\n==== 平均每轮耗时(ms) ====" << std::endl;
        std::cout << "preprocess: " << total_pre / rounds << " (" << (total_pre / sum * 100) << "%)" << std::endl;
        std::cout << "H2D:        " << total_h2d / rounds << " (" << (total_h2d / sum * 100) << "%)" << std::endl;
        std::cout << "infer:      " << total_infer / rounds << " (" << (total_infer / sum * 100) << "%)" << std::endl;
        std::cout << "D2H:        " << total_d2h / rounds << " (" << (total_d2h / sum * 100) << "%)" << std::endl;
        std::cout << "post:       " << total_post / rounds << " (" << (total_post / sum * 100) << "%)" << std::endl;
        std::cout << "NMS:        " << total_nms / rounds << " (" << (total_nms / sum * 100) << "%)" << std::endl;
        std::cout << "总计:       " << sum / rounds << " (100%)" << std::endl;
    }
}

BENCHMARK(BM_TensorRT_FullPipeline);

BENCHMARK_MAIN();