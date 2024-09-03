#include "opencvinferenceengine.h"

#include <iostream> // or any log system
namespace cvedia {
namespace { // constants values
	constexpr double SCALE_FACTOR{ 1.0 / 255.0 };
	const cv::Size IMAGE_INPUT_SIZE(416, 416);
	constexpr float THRESHOLD_OBJECTNESS{ 0.5f };
	constexpr float THRESHOLD_SCORE{ 0.55f };
	constexpr float THRESHOLD_NON_MAXIMUM_SUPRESSION{ 0.4f };
}

void OpenCVInferenceEngine::loadModel(const std::string& modelConfig, const std::string& modelWeights) {
	m_network = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
	
	// TODO: pass these target by parameter
	m_network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	m_network.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

	if (m_network.empty()) {
		std::cerr << "Could not load the network!" << std::endl;
	}

	// determine output layers
	std::vector<int> indexesOfOutputLayers{ m_network.getUnconnectedOutLayers() };
	std::vector<std::string> layerNames{ m_network.getLayerNames() };
	for (size_t i = 0; i < indexesOfOutputLayers.size(); i++) {
		const int index{ indexesOfOutputLayers[i] - 1 };
		namesOfOutputLayers.emplace_back(layerNames[index]);
	}
}

core::ObjectDetectionResult OpenCVInferenceEngine::inference(const cv::Mat& frame) {
	// convert to blob
	const cv::Mat inputBlob{ cv::dnn::blobFromImage(frame, SCALE_FACTOR, IMAGE_INPUT_SIZE, cv::Scalar(), true, false) };
	m_network.setInput(inputBlob);
	std::vector<cv::Mat> outputs;
	m_network.forward(outputs, namesOfOutputLayers);	// TODO: maybe not need all layers

	core::ObjectDetectionResult detectionObjectResult;
	
	cv::Mat row, scores;
	double maxClassConfidence;
	cv::Point classIdPoint;

	// check all output layers and their results to find the max confidence match to be stored
	for (const auto& output : outputs) {
		for (int index = 0; index < output.rows; index++) {
			row = output.row(index);
			scores = row.colRange(5, output.cols);

			// find the class with the highest confidence		
			cv::minMaxLoc(scores, nullptr, &maxClassConfidence, nullptr, &classIdPoint);

			const float objectness = row.at<float>(4);
			if (objectness > THRESHOLD_OBJECTNESS) {
				const int centerX = static_cast<int>(row.at<float>(0) * frame.cols);
				const int centerY = static_cast<int>(row.at<float>(1) * frame.rows);
				const int width = static_cast<int>(row.at<float>(2) * frame.cols);
				const int height = static_cast<int>(row.at<float>(3) * frame.rows);
				const int left = static_cast<int>(centerX - width * 0.5f);
				const int top = static_cast<int>(centerY - height * 0.5f);
				detectionObjectResult.boundingBoxes.emplace_back(cv::Rect(left, top, width, height));
				detectionObjectResult.classIds.emplace_back(classIdPoint.x);
				detectionObjectResult.confidences.emplace_back(static_cast<float>(maxClassConfidence));
			}
		}
	}

	// apply NMS
	cv::dnn::NMSBoxes(detectionObjectResult.boundingBoxes, 
		detectionObjectResult.confidences, 
		THRESHOLD_SCORE, 
		THRESHOLD_NON_MAXIMUM_SUPRESSION, 
		detectionObjectResult.indices);

	return detectionObjectResult;
}

// To use future in the main code, just left this unused function
std::future<core::ObjectDetectionResult> OpenCVInferenceEngine::asyncInference(const cv::Mat& frame) {
	return std::async(std::launch::async, &OpenCVInferenceEngine::inference, this, frame);
}
};