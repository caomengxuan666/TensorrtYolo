#pragma once

#ifdef TENSORRTYOLO_EXPORTS
#define CMX_API __declspec(dllexport)
#else
#define CMX_API __declspec(dllimport)
#endif

#include "Detection.h"
#include <functional>
#include <memory>
#include <vector>

/**
 * @brief Callback function type for custom visualization
 *
 * @param bgrData Pointer to BGR image data
 * @param width Width of the image
 * @param height Height of the image
 * @param detections Vector of detected objects
 * @param userData User-defined data passed with the callback
 */
using VisualizationFunction = std::function<void(
    const unsigned char *bgrData, int width, int height, // 增加const
    const std::vector<Detection> &detections, void *userData)>;
class DetectorConfig;

/**
 * @brief Main API class for TensorRT YOLO Detector
 *
 * This class provides an interface for performing object detection using a YOLO
 * model accelerated with NVIDIA TensorRT. It supports both basic and custom
 * visualization.
 */
class CMX_API TensorrtYoloDetectorAPI {
public:
  /**
   * @brief Construct a new Tensorrt Yolo Detector API object
   *
   * @param engineFile Path to the Config file
   */
  explicit TensorrtYoloDetectorAPI(const DetectorConfig &config);

  /**
   * @brief Destroy the Tensorrt Yolo Detector API object
   */
  ~TensorrtYoloDetectorAPI();

  /**
   * @brief Perform object detection on an image
   *
   * @param bgrData Pointer to BGR image data
   * @param width Width of the image
   * @param height Height of the image
   * @return std::vector<Detection> Vector of detected objects
   */
  std::vector<Detection> detect(const unsigned char *bgrData, int width,
                                int height);

  /**
   * @brief Perform object classification on an image
   *
   * @param bgrData Pointer to BGR image data
   * @param width Width of the image
   * @param height Height of the image
   * @return std::vector<Classification> Vector of classified objects
   */
  std::vector<Classification> classify(const unsigned char *bgrData, int width,
                                       int height);
  /**
   * @brief Visualize detection results on an image (basic visualization)
   *
   * @param bgrData Pointer to BGR image data
   * @param width Width of the image
   * @param height Height of the image
   * @param detections Vector of detected objects
   */
  void visualize(unsigned char *bgrData, int width, int height,
                 const std::vector<Detection> &detections);

  /**
   * @brief Visualize detection results on an image (basic visualization)
   *
   * @param bgrData Pointer to BGR image data
   * @param width Width of the image
   * @param height Height of the image
   * @param classifications Vector of detected classifications
   */
  void visualize(unsigned char *bgrData, int width, int height,
                 const std::vector<Classification> &classifications);

  // 注册自定义可视化函数
  /**
   * @brief Register a custom visualization callback function
   * @note This only works for Detection results, not classifications.
   * @param callback The callback function to register
   * @param userData Optional user-defined data to pass with the callback
   */
  void registerVisualizationCallback(VisualizationFunction callback,
                                     void *userData = nullptr);

  // 使用自定义回调的可视化方法
  /**
   * @brief Visualize detection results using the registered custom callback
   * @note This only works for Detection results, not classifications.
   * @param bgrData Pointer to BGR image data
   * @param width Width of the image
   * @param height Height of the image
   * @param detections Vector of detected objects
   */
  void visualizeWithCallback(unsigned char *bgrData, int width, int height,
                             const std::vector<Detection> &detections);

  TensorrtYoloDetectorAPI(const TensorrtYoloDetectorAPI &) = delete;
  TensorrtYoloDetectorAPI &operator=(const TensorrtYoloDetectorAPI &) = delete;

private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
};