#ifndef __INCLUDE_GUIENGINE_H__
#define __INCLUDE_GUIENGINE_H__

#include "detectedobject.h"
namespace cvedia {

// to display the video side by side the result
class GuiEngine {
public:
	virtual ~GuiEngine() = default;
	virtual bool init(int count, ...) = 0;
	virtual void display(const cv::Mat& original, const cv::Mat& frame, const core::ObjectDetectionResult& result) = 0;
	virtual bool update() = 0;
	virtual void release() = 0;
};
};
#endif // !__INCLUDE_GUIENGINE_H__