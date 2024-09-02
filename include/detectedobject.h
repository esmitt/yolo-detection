#ifndef __INCLUDE_DETECTEDOBJECT__H
#define __INCLUDE_DETECTEDOBJECT__H

#include <opencv2/core/types.hpp>

#include <cstdarg>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cvedia {
namespace core {
using BoundingBoxes = std::vector<cv::Rect>;
using VecColors = std::vector<cv::Scalar>;

// to store the result of the inference
struct ObjectDetectionResult {
	BoundingBoxes boundingBoxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	VecColors colors;
	std::vector<int> indices;	// this has values if NMS is applied
};
};
};
#endif // !__INCLUDE_DETECTEDOBJECT__H
