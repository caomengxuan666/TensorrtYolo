#include "DetectorConfig.h"
#include "TensorrtYoloDetectorAPI.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    try {
        // 配置参数（可从文件加载）
        DetectorConfig config;
        config.setEngineFile("best.engine")
            .setNumClasses(1)
            .setConfidenceThreshold(0.3f)
            .setNMSThreshold(0.5f);

        // 创建检测器实例
        TensorrtYoloDetectorAPI detector(config);

        // 读取测试图像
        cv::Mat image = cv::imread("test_image.png");
        if (image.empty()) {
            std::cerr << "无法读取图像，请检查图像路径!" << std::endl;
            return -1;
        }

        // 执行目标检测
        std::vector<Detection> results = detector.detect(image.data, image.cols, image.rows);

        // 可视化结果
        detector.visualize(image.data, image.cols, image.rows, results);

        // 显示结果图像
        cv::imshow("YOLO Visualization", image);
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}