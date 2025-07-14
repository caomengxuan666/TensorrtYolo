// Config.h
#pragma once
#include <string>
#include <vector>

class DetectorConfig {
public:
    // 默认构造函数
    DetectorConfig() = default;

    // 从文件加载配置（可选）
    explicit DetectorConfig(const std::string& configPath) {
        loadFromFile(configPath);
    }

    // 设置配置参数
    DetectorConfig& setEngineFile(const std::string& path) {
        engineFile_ = path;
        return *this;
    }

    DetectorConfig& setNumClasses(int num) {
        numClasses_ = num;
        return *this;
    }

    DetectorConfig& setConfidenceThreshold(float thresh) {
        confThresh_ = thresh;
        return *this;
    }

    DetectorConfig& setNMSThreshold(float thresh) {
        nmsThresh_ = thresh;
        return *this;
    }

    // 从文件加载配置（简化实现，实际项目可扩展）
    bool loadFromFile(const std::string& configPath) {
        // 示例：从JSON/YAML读取配置
        // 实际项目中可使用第三方库（如nlohmann/json）
        // 此处简化为硬编码示例
        engineFile_ = "best.engine";
        numClasses_ = 80;
        confThresh_ = 0.25f;
        nmsThresh_ = 0.45f;
        return true;
    }

    // 获取配置参数
    const std::string& getEngineFile() const { return engineFile_; }
    int getNumClasses() const { return numClasses_; }
    float getConfidenceThreshold() const { return confThresh_; }
    float getNMSThreshold() const { return nmsThresh_; }

private:
    std::string engineFile_="best.engine";    // TensorRT引擎文件路径
    int numClasses_ = 1;       // 类别数量
    float confThresh_ = 0.4f;  // 置信度阈值
    float nmsThresh_ = 0.45f;   // NMS阈值
};