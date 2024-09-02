#include "opencvguiengine.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

namespace cvedia {

// 1st parameter is the class filename, the 2nd is the color filename
bool OpenCVGuiEngine::init(int count, ...) {
	if (count == 2) {
		std::va_list args;
		va_start(args, count);
		const std::string classNameFilename = va_arg(args, std::string);
		const std::string classColorFilename = va_arg(args, std::string);
		va_end(args);
		loadClassesNames(classNameFilename);
		loadColors(classColorFilename);
		if (m_classesNames.size() == 0 || m_classesColors.size() == 0) {
			std::cerr << "Filenames of class name or color name are empty!" << std::endl;
			return false;
		}
	}
	else {
		return false;
	}
	return true;
}

void OpenCVGuiEngine::display(const cv::Mat& original, const cv::Mat& frame, const core::ObjectDetectionResult& result) {
	cv::Mat combined;
	for (size_t i = 0; i < result.indices.size(); i++) {
		const int index = result.indices[i];
		const cv::Rect& box = result.boundingBoxes[index];
		const auto id = result.classIds[index];
		const auto classId = result.classIds[index];

		if (classId < 0 || classId >= static_cast<int>(m_classesColors.size())) {
			std::cerr << "Invalid classId: " << classId << std::endl;
			continue;
		}
		
		// draw bounding box
		const cv::Scalar color = m_classesColors[id];
		cv::rectangle(frame, box, color, 2);

		const std::string label = m_classesNames[classId] + " <" + cv::format("%.2f", result.confidences[index]) + ">";

		// draw the name of the class with the confidence value
		int baseLine;
		const cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
		const int top = std::max(box.y, labelSize.height);
		putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
	}

	cv::hconcat(original, frame, combined);
	cv::imshow("CVEDIA C++ Tech Test", combined);
}

bool OpenCVGuiEngine::update() {
	if (cv::waitKey(1) == 'q') {
		return false;
	}
	return true;
}


void OpenCVGuiEngine::loadClassesNames(const std::string& filename) {
	std::ifstream ifs{ filename };
	if (!ifs) {
		throw std::runtime_error("Failed to open file: " + filename);
	}
	std::string line;
	while (std::getline(ifs, line)) {
		m_classesNames.emplace_back(std::move(line));
	}
}

void OpenCVGuiEngine::loadColors(const std::string& filename) {
	std::ifstream ifs{ filename };
	if (!ifs) {
		throw std::runtime_error("Failed to open file: " + filename);
	}
	std::vector<cv::Scalar> colors;
	std::string line;
	while (std::getline(ifs, line)) {
		line.erase(std::remove(line.begin(), line.end(), ','), line.end());
		std::istringstream iss(line);

		int r, g, b;
		if (iss >> r >> g >> b) {
			m_classesColors.push_back(cv::Scalar(b, g, r));
		}
		else {
			std::cerr << "Error parsing line: " << line << std::endl;
		}
	}
}

void OpenCVGuiEngine::release() {
	cv::destroyAllWindows();
}
}
