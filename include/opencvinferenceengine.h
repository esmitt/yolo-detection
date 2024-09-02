#ifndef __INCLUDE_OPENCVINFERENCEENGINE__H
#define __INCLUDE_OPENCVINFERENCEENGINE__H

#include "inferenceengine.h"

#include <opencv2/core/types.hpp>

#include <opencv2/dnn/dnn.hpp>

#include <string>
#include <vector>

// source: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
namespace cvedia {
class OpenCVInferenceEngine : public InferenceEngine {
public:
  OpenCVInferenceEngine() = default;

  void loadModel(const std::string& modelConfig, const std::string& modelWeights) override;
  core::ObjectDetectionResult inference(const cv::Mat& frame) override;
  [[maybe_unused]] std::future<core::ObjectDetectionResult> asyncInference(const cv::Mat& frame) override;
private:
  cv::dnn::Net m_network;
  std::vector<std::string> namesOfOutputLayers;
};
};
#endif // !__INCLUDE_OPENCVINFERENCEENGINE__H