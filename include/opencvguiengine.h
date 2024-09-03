#ifndef __INCLUDE_OPENCVGUIENGINE_H__
#define __INCLUDE_OPENCVGUIENGINE_H__

#include "guiengine.h"

namespace cvedia {

// Particular implementation using the basic GUI of OpenCV
class OpenCVGuiEngine : public GuiEngine {
public:
	bool init(int count, ...) override;
	void display(const cv::Mat& original, const cv::Mat& frame, const core::ObjectDetectionResult& result) override;
	bool update() override;
	void release() override;

private:
	void loadClassesNames(const std::string& filename);
	void loadColors(const std::string& filename);

	std::vector<cv::Scalar> m_classesColors;
	std::vector<std::string> m_classesNames;
};
};
#endif // !__INCLUDE_OPENCVGUIENGINE_H__
