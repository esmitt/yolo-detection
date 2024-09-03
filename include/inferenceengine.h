#ifndef __INCLUDE_INFERENCEENGINE__H
#define __INCLUDE_INFERENCEENGINE__H

#include <opencv2/core/types.hpp>

#include "detectedobject.h"

#include <future>
#include <string>
#include <vector>

namespace cvedia {

// To use a different inference Engine
class InferenceEngine {
public:
  virtual ~InferenceEngine() = default;
  virtual void loadModel(const std::string& modelConfig, const std::string& modelWeights) = 0;
  virtual core::ObjectDetectionResult inference(const cv::Mat& frame) = 0;
  [[maybe_unused]] virtual std::future<core::ObjectDetectionResult> asyncInference(const cv::Mat& frame) = 0;
};
};
#endif // !__INCLUDE_INFERENCEENGINE__H