// 抑制C4251警告（仅针对当前类）
#pragma warning(push)
#pragma warning(disable : 4251)

#include "TensorrtYoloDetectorAPI.h"
#include "DetectorConfig.h"
// 第三方依赖仅在cpp中包含
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <execution>
#include <fstream>
#include <future>
#include <opencv2/opencv.hpp>
#include <stdexcept>

using namespace nvinfer1;

struct InternalDetection {
    int classId;
    float score;
    cv::Rect box;
    bool operator<(const InternalDetection& other) const { return score > other.score; }
};

// -------------------------- 内部功能类（单一职责） --------------------------

// 1. 日志管理类（仅负责TensorRT日志输出）
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

// 2. 检测结果转换类（仅负责InternalDetection与Detection的转换）
class DetectionConverter {
public:
    // Detection -> InternalDetection（含OpenCV Rect）
    static std::vector<InternalDetection> toInternal(const std::vector<Detection>& detections)
    {
        std::vector<InternalDetection> result;
        result.reserve(detections.size());
        for (const auto& det : detections) {
            result.push_back({ det.class_id,
                det.score,
                cv::Rect(det.left, det.top, det.width, det.height) });
        }
        return result;
    }

    // InternalDetection -> Detection（剥离OpenCV依赖）
    static std::vector<Detection> fromInternal(const std::vector<InternalDetection>& internalDets)
    {
        std::vector<Detection> result;
        result.reserve(internalDets.size());
        for (const auto& det : internalDets) {
            result.push_back({ det.classId,
                det.score,
                det.box.x,
                det.box.y,
                det.box.width,
                det.box.height });
        }
        return result;
    }
};

// 3. TensorRT引擎管理类（仅负责引擎加载、推理执行、CUDA资源管理）
class TensorrtEngineManager {
public:
    TensorrtEngineManager(Logger& logger, const std::string& engineFile)
        : logger_(logger)
        , input_w_(640)
        , input_h_(640)
        , num_boxes_(8400)
    {
        loadEngine(engineFile);
        initCudaResources();
    }

    ~TensorrtEngineManager()
    {
        if (d_input_)
            cudaFree(d_input_);
        if (d_output_)
            cudaFree(d_output_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    // 执行推理（输入预处理后的数据，输出模型原始输出）
    void infer(const std::vector<float>& inputData, std::vector<float>& outputData)
    {
        // 主机到设备的数据拷贝
        if (cudaMemcpyAsync(d_input_, inputData.data(), inputSize_,
                cudaMemcpyHostToDevice, stream_)
            != cudaSuccess) {
            throw std::runtime_error("Failed to copy input to device");
        }

        // 设置输入输出地址并执行推理
        context_->setInputTensorAddress(inputName_.c_str(), d_input_);
        context_->setOutputTensorAddress(outputName_.c_str(), d_output_);
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("Inference enqueue failed");
        }

        // 设备到主机的数据拷贝
        outputData.resize(5 * num_boxes_);
        if (cudaMemcpyAsync(outputData.data(), d_output_, outputSize_,
                cudaMemcpyDeviceToHost, stream_)
            != cudaSuccess) {
            throw std::runtime_error("Failed to copy output from device");
        }
        cudaStreamSynchronize(stream_);
    }

    // 获取模型输入尺寸（供预处理使用）
    int inputWidth() const { return input_w_; }
    int inputHeight() const { return input_h_; }

    // 获取输出框数量（供后处理使用）
    int numBoxes() const { return num_boxes_; }

private:
    // 加载引擎文件
    void loadEngine(const std::string& engineFile)
    {
        // 1. 检查文件路径是否为空
        if (engineFile.empty()) {
            throw std::invalid_argument("Engine file path is empty (invalid path)");
        }

        // 2. 检查文件是否存在且可打开
        std::ifstream file(engineFile, std::ios::binary | std::ios::ate); // 打开并定位到文件末尾
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open engine file: " + engineFile + " (check if path is correct or file exists)");
        }

        // 3. 检查文件大小（避免空文件或损坏文件）
        std::streamsize fileSize = file.tellg(); // 获取文件大小
        if (fileSize <= 0) {
            file.close();
            throw std::runtime_error("Engine file is empty or corrupted: " + engineFile);
        }

        // 4. 读取文件数据（移动到文件开头）
        file.seekg(0, std::ios::beg);
        std::vector<char> engineData(fileSize);
        if (!file.read(engineData.data(), fileSize)) {
            file.close();
            throw std::runtime_error("Failed to read engine file: " + engineFile + " (file may be corrupted or read permission denied)");
        }
        file.close();

        // 5. 初始化TensorRT组件（增强错误信息）
        runtime_.reset(createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime (check TensorRT installation)");
        }

        // 6. 反序列化引擎（最常见的失败点，明确提示文件损坏）
        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize engine from file: " + engineFile + " (file may be corrupted, incompatible with current TensorRT version, or not a valid engine file)");
        }

        // 7. 创建推理上下文
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create inference context (insufficient GPU memory or invalid engine)");
        }

        // 8. 解析输入输出信息（失败时提示引擎结构异常）
        try {
            parseIOInfo();
        } catch (...) {
            throw std::runtime_error("Failed to parse input/output tensors from engine (engine file may have invalid structure)");
        }
    }

    // 解析输入输出名称和尺寸
    void parseIOInfo()
    {
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                inputName_ = name;
                // 假设输入尺寸为3x640x640（可根据实际引擎调整）
            } else {
                outputName_ = name;
            }
        }
    }

    // 初始化CUDA资源（流、设备内存）
    void initCudaResources()
    {
        inputSize_ = 3 * input_h_ * input_w_ * sizeof(float);
        outputSize_ = 5 * num_boxes_ * sizeof(float);

        if (cudaMalloc(&d_input_, inputSize_) != cudaSuccess)
            throw std::runtime_error("Failed to allocate input memory");
        if (cudaMalloc(&d_output_, outputSize_) != cudaSuccess)
            throw std::runtime_error("Failed to allocate output memory");
        if (cudaStreamCreate(&stream_) != cudaSuccess)
            throw std::runtime_error("Failed to create cuda stream");
    }

    // TensorRT组件（智能指针管理生命周期）
    using RuntimePtr = std::unique_ptr<IRuntime, decltype([](IRuntime* p) { delete p; })>;
    using EnginePtr = std::unique_ptr<ICudaEngine, decltype([](ICudaEngine* p) { delete p; })>;
    using ContextPtr = std::unique_ptr<IExecutionContext, decltype([](IExecutionContext* p) { delete p; })>;

    RuntimePtr runtime_;
    EnginePtr engine_;
    ContextPtr context_;
    Logger& logger_; // 引用外部日志器（不负责生命周期）
    std::string inputName_, outputName_;
    int input_w_, input_h_, num_boxes_;
    size_t inputSize_, outputSize_;
    void* d_input_ = nullptr; // 设备端输入内存
    void* d_output_ = nullptr; // 设备端输出内存
    cudaStream_t stream_ = nullptr;
};

// 4. 图像预处理类（仅负责图像缩放、填充、归一化）
class ImagePreprocessor {
public:
    // 预处理：将原始图像转换为模型输入格式
    void preprocess(const unsigned char* bgrData, int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        std::vector<float>& outputData,
        float& scale, int& dw, int& dh)
    {
        // 转换为OpenCV矩阵（不拷贝数据，仅包装指针）
        cv::Mat img(srcHeight, srcWidth, CV_8UC3, const_cast<unsigned char*>(bgrData));

        // 计算缩放比例和填充尺寸
        scale = std::min((float)dstWidth / srcWidth, (float)dstHeight / srcHeight);
        int newW = srcWidth * scale;
        int newH = srcHeight * scale;
        dw = (dstWidth - newW) / 2; // 水平填充
        dh = (dstHeight - newH) / 2; // 垂直填充

        // 缩放并填充图像
        cv::Mat resized, padded(dstHeight, dstWidth, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::resize(img, resized, cv::Size(newW, newH));
        resized.copyTo(padded(cv::Rect(dw, dh, newW, newH)));

        // 并行归一化（转换为float并归一化到[0,1]）
        outputData.resize(3 * dstHeight * dstWidth);
        for (int c = 0; c < 3; ++c) { // BGR三通道
            std::for_each(std::execution::par,
                padded.begin<cv::Vec3b>(), padded.end<cv::Vec3b>(),
                [&](cv::Vec3b& pixel) {
                    int y = &pixel - padded.ptr<cv::Vec3b>(0); // 计算行索引
                    int x = y % dstHeight; // 计算列索引
                    y /= dstHeight;
                    outputData[c * dstHeight * dstWidth + y * dstWidth + x] = pixel[c] / 255.0f;
                });
        }
    }
};

// 5. 后处理类（仅负责解析模型输出、过滤、NMS）
class DetectionPostprocessor {
public:
    explicit DetectionPostprocessor(const DetectorConfig& config)
        : num_classes_(config.getNumClasses())
        , conf_thresh_(config.getConfidenceThreshold())
        , nms_thresh_(config.getNMSThreshold())
    {
    }
    // 后处理主逻辑：解析输出->过滤低置信度->NMS->坐标转换
    std::vector<InternalDetection> process(const std::vector<float>& outputData,
        int srcWidth, int srcHeight,
        float scale, int dw, int dh,
        int numBoxes)
    {
        // 解析模型输出（过滤低置信度框）
        std::vector<InternalDetection> detections;
        detections.reserve(numBoxes);
        for (int i = 0; i < numBoxes; ++i) {
            float cx = outputData[i];
            float cy = outputData[numBoxes + i];
            float w = outputData[2 * numBoxes + i];
            float h = outputData[3 * numBoxes + i];
            float score = outputData[4 * numBoxes + i];

            if (score < conf_thresh_)
                continue; // 过滤低置信度

            // 转换坐标到原始图像（去缩放和填充）
            int x1 = (cx - w / 2 - dw) / scale;
            int y1 = (cy - h / 2 - dh) / scale;
            int x2 = (cx + w / 2 - dw) / scale;
            int y2 = (cy + h / 2 - dh) / scale;

            // 裁剪到图像边界
            x1 = std::max(0, std::min(x1, srcWidth - 1));
            y1 = std::max(0, std::min(y1, srcHeight - 1));
            x2 = std::max(0, std::min(x2, srcWidth - 1));
            y2 = std::max(0, std::min(y2, srcHeight - 1));

            detections.push_back({ 0, score, cv::Rect(x1, y1, x2 - x1, y2 - y1) });
        }

        // 执行NMS
        return nonMaximumSuppression(detections);
    }

private:
    // 非极大值抑制（NMS）
    std::vector<InternalDetection> nonMaximumSuppression(std::vector<InternalDetection>& detections)
    {
        if (detections.empty())
            return {};

        // 按置信度降序排序
        std::sort(std::execution::par, detections.begin(), detections.end(),
            [](const InternalDetection& a, const InternalDetection& b) {
                return a.score > b.score;
            });

        // 计算所有框的面积
        std::vector<float> areas(detections.size());
        std::transform(std::execution::par, detections.begin(), detections.end(), areas.begin(),
            [](const InternalDetection& d) {
                return (float)(d.box.width * d.box.height);
            });

        std::vector<bool> suppressed(detections.size(), false);
        std::vector<InternalDetection> result;

        // 执行NMS
        for (size_t i = 0; i < detections.size(); ++i) {
            if (suppressed[i])
                continue;
            result.push_back(detections[i]);

            // 并行计算与后续框的IOU，标记重叠度过高的框
            const auto& currentBox = detections[i].box;
            const float currentArea = areas[i];
            std::for_each(std::execution::par_unseq,
                detections.begin() + i + 1, detections.end(),
                [&](const InternalDetection& d) {
                    size_t j = &d - &detections[0];
                    if (suppressed[j])
                        return;

                    // 计算IOU
                    int xx1 = std::max(currentBox.x, d.box.x);
                    int yy1 = std::max(currentBox.y, d.box.y);
                    int xx2 = std::min(currentBox.x + currentBox.width, d.box.x + d.box.width);
                    int yy2 = std::min(currentBox.y + currentBox.height, d.box.y + d.box.height);
                    int interArea = std::max(0, xx2 - xx1) * std::max(0, yy2 - yy1);
                    float iou = interArea / (currentArea + areas[j] - interArea);

                    if (iou > nms_thresh_)
                        suppressed[j] = true;
                });
        }
        return result;
    }

private:
    int num_classes_;
    float conf_thresh_;
    float nms_thresh_;
};

// 6. 可视化类（仅负责绘制检测结果和管理回调）
class DetectionVisualizer {
public:
    // 注册自定义可视化回调
    void setCallback(VisualizationFunction callback, void* userData)
    {
        callback_ = callback;
        userData_ = userData;
    }

    // 绘制内部检测结果（使用OpenCV，美化样式）
    void drawInternal(const unsigned char* bgrData, int width, int height,
        const std::vector<InternalDetection>& detections)
    {
        cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(bgrData));

        // 统一颜色 (绿色)
        cv::Scalar color(0, 255, 0);

        // 自定义字体和样式
        const int fontFace = cv::FONT_HERSHEY_DUPLEX;
        const float fontScale = 0.6;
        const int thickness = 1;
        const int radius = 8; // 圆角半径

        for (const auto& d : detections) {
            cv::Rect box = d.box;

            // 绘制圆角边框
            // 绘制四条边
            cv::line(img, cv::Point(box.x + radius, box.y),
                cv::Point(box.x + box.width - radius, box.y),
                color, 2, cv::LINE_AA);
            cv::line(img, cv::Point(box.x + box.width, box.y + radius),
                cv::Point(box.x + box.width, box.y + box.height - radius),
                color, 2, cv::LINE_AA);
            cv::line(img, cv::Point(box.x + radius, box.y + box.height),
                cv::Point(box.x + box.width - radius, box.y + box.height),
                color, 2, cv::LINE_AA);
            cv::line(img, cv::Point(box.x, box.y + radius),
                cv::Point(box.x, box.y + box.height - radius),
                color, 2, cv::LINE_AA);

            // 绘制四个圆角
            cv::ellipse(img, cv::Point(box.x + radius, box.y + radius),
                cv::Size(radius, radius), 180, 0, 90, color, 2, cv::LINE_AA);
            cv::ellipse(img, cv::Point(box.x + box.width - radius, box.y + radius),
                cv::Size(radius, radius), 270, 0, 90, color, 2, cv::LINE_AA);
            cv::ellipse(img, cv::Point(box.x + radius, box.y + box.height - radius),
                cv::Size(radius, radius), 90, 0, 90, color, 2, cv::LINE_AA);
            cv::ellipse(img, cv::Point(box.x + box.width - radius, box.y + box.height - radius),
                cv::Size(radius, radius), 0, 0, 90, color, 2, cv::LINE_AA);

            // 准备标签文本（保留两位小数）
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2);
            ss << "Class: " << d.classId << " Score: " << d.score;
            std::string label = ss.str();

            // 计算文本尺寸
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

            // 绘制文本背景（带圆角）
            cv::Rect textBg(box.x, box.y - textSize.height - baseline - 4,
                textSize.width + 8, textSize.height + baseline + 4);
            cv::rectangle(img, textBg, color, -1, cv::LINE_AA);

            // 绘制文本
            cv::putText(img, label,
                cv::Point(box.x + 4, box.y - 4),
                fontFace, fontScale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
        }
    }

    // 绘制外部检测结果（使用OpenCV，带圆角边框等样式）
    void drawExternal(const unsigned char* bgrData, int width, int height,
        const std::vector<Detection>& detections)
    {
        // 关键修复：使用const_cast移除const属性，适配cv::Mat的构造函数
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char*>(bgrData));

        // 统一颜色 (蓝色)
        cv::Scalar color(255, 128, 0);

        // 自定义字体和样式
        const int fontFace = cv::FONT_HERSHEY_DUPLEX;
        const float fontScale = 0.6;
        const int thickness = 1;

        for (const auto& det : detections) {
            // 绘制圆角边框
            cv::Rect box(det.left, det.top, det.width, det.height);
            int radius = std::min(box.width, box.height) / 8;

            // 绘制四条边
            cv::line(image, cv::Point(box.x + radius, box.y),
                cv::Point(box.x + box.width - radius, box.y),
                color, 2, cv::LINE_AA);
            cv::line(image, cv::Point(box.x + box.width, box.y + radius),
                cv::Point(box.x + box.width, box.y + box.height - radius),
                color, 2, cv::LINE_AA);
            cv::line(image, cv::Point(box.x + radius, box.y + box.height),
                cv::Point(box.x + box.width - radius, box.y + box.height),
                color, 2, cv::LINE_AA);
            cv::line(image, cv::Point(box.x, box.y + radius),
                cv::Point(box.x, box.y + box.height - radius),
                color, 2, cv::LINE_AA);

            // 绘制四个圆角
            cv::ellipse(image, cv::Point(box.x + radius, box.y + radius),
                cv::Size(radius, radius), 180, 0, 90, color, 2, cv::LINE_AA);
            cv::ellipse(image, cv::Point(box.x + box.width - radius, box.y + radius),
                cv::Size(radius, radius), 270, 0, 90, color, 2, cv::LINE_AA);
            cv::ellipse(image, cv::Point(box.x + radius, box.y + box.height - radius),
                cv::Size(radius, radius), 90, 0, 90, color, 2, cv::LINE_AA);
            cv::ellipse(image, cv::Point(box.x + box.width - radius, box.y + box.height - radius),
                cv::Size(radius, radius), 0, 0, 90, color, 2, cv::LINE_AA);

            // 准备标签文本
            std::string label = "Class: " + std::to_string(det.class_id) + " Score: " + std::to_string(int(det.score * 100)) + "%";

            // 计算文本尺寸
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

            // 绘制文本背景
            cv::rectangle(image,
                cv::Point(box.x, box.y - textSize.height - baseline),
                cv::Point(box.x + textSize.width, box.y),
                color, -1, cv::LINE_AA);

            // 绘制文本
            cv::putText(image, label,
                cv::Point(box.x, box.y - baseline),
                fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }

    // 使用回调或默认可视化
    void visualizeWithCallback(const unsigned char* bgrData, int width, int height,
        const std::vector<Detection>& detections)
    {
        if (callback_) {
            callback_(bgrData, width, height, detections, userData_);
        } else {
            // 回调未注册时，使用内部转换后的结果绘制
            auto internal = DetectionConverter::toInternal(detections);
            drawInternal(bgrData, width, height, internal);
        }
    }

private:
    VisualizationFunction callback_; // 自定义回调
    void* userData_ = nullptr; // 回调用户数据
};

// -------------------------- 核心实现类（协调各模块） --------------------------

// Impl类：协调上述6个功能类，不直接实现具体功能
class TensorrtYoloDetectorAPI::Impl {
public:
    // 接收配置类而非硬编码参数
    explicit Impl(const DetectorConfig& config)
        : logger_()
        , engineManager_(logger_, config.getEngineFile())
        , preprocessor_()
        , postprocessor_(config)
        , visualizer_() // 直接传入配置类
        , config_(config)
    {
    }

    // 检测流程：预处理->推理->后处理->转换结果
    std::vector<InternalDetection> detect(const unsigned char* bgrData, int width, int height)
    {
        // 1. 预处理
        float scale;
        int dw, dh;
        std::vector<float> inputData;
        preprocessor_.preprocess(bgrData, width, height,
            engineManager_.inputWidth(), engineManager_.inputHeight(),
            inputData, scale, dw, dh);

        // 2. 推理
        std::vector<float> outputData;
        engineManager_.infer(inputData, outputData);

        // 3. 后处理
        return postprocessor_.process(outputData, width, height, scale, dw, dh,
            engineManager_.numBoxes());
    }

    // 可视化相关（转发给visualizer_）
    void visualize(unsigned char* bgrData, int width, int height,
        const std::vector<InternalDetection>& detections)
    {
        visualizer_.drawInternal(bgrData, width, height, detections);
    }

    void registerVisualizationCallback(VisualizationFunction callback, void* userData)
    {
        visualizer_.setCallback(callback, userData);
    }

    void visualizeWithCallback(unsigned char* bgrData, int width, int height,
        const std::vector<Detection>& detections)
    {
        visualizer_.visualizeWithCallback(bgrData, width, height, detections);
    }

private:
    // 聚合各功能类（职责单一，由Impl协调）
    Logger logger_;
    TensorrtEngineManager engineManager_; // 引擎管理
    ImagePreprocessor preprocessor_; // 预处理
    DetectionPostprocessor postprocessor_; // 后处理
    DetectionVisualizer visualizer_; // 可视化
    DetectorConfig config_; // 配置参数（可从文件加载）
};

// -------------------------- API方法实现（转发给Impl） --------------------------

TensorrtYoloDetectorAPI::TensorrtYoloDetectorAPI(const DetectorConfig& config)
    : pImpl_(std::make_unique<Impl>(config))
{
}

TensorrtYoloDetectorAPI::~TensorrtYoloDetectorAPI() = default;

std::vector<Detection> TensorrtYoloDetectorAPI::detect(const unsigned char* bgrData, int width, int height)
{
    auto internalDets = pImpl_->detect(bgrData, width, height);
    return DetectionConverter::fromInternal(internalDets); // 转换为外部结果
}

void TensorrtYoloDetectorAPI::visualize(unsigned char* bgrData, int width, int height,
    const std::vector<Detection>& detections)
{
    auto internal = DetectionConverter::toInternal(detections);
    pImpl_->visualize(bgrData, width, height, internal);
}

void TensorrtYoloDetectorAPI::registerVisualizationCallback(VisualizationFunction callback, void* userData)
{
    pImpl_->registerVisualizationCallback(callback, userData);
}

void TensorrtYoloDetectorAPI::visualizeWithCallback(unsigned char* bgrData, int width, int height,
    const std::vector<Detection>& detections)
{
    pImpl_->visualizeWithCallback(bgrData, width, height, detections);
}