#include "inferenceengine.h"
#include "opencvinferenceengine.h"
#include "opencvguiengine.h"
#include "guiengine.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <filesystem>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <queue>

namespace user_input {
	std::string program_input(int argc, char* argv[]) {
		if (argc > 2) {
			throw std::runtime_error("too many input parameters!");
		}
		if (argc == 1) {
			throw std::invalid_argument("need the name of the video file");
		}
		
		const std::string_view filename(argv[1]);
		std::cout << filename << std::endl;
		if (!std::filesystem::exists(filename)) {
			throw std::runtime_error(": No such file");
		}
		return filename.data();
	}
}

namespace reading_video {
	std::queue<cv::Mat> frameQueue;
	std::mutex queueMutex;
	bool keepRunning = true;

	void readFrames(cv::VideoCapture& cap) {
		while (keepRunning) {
			cv::Mat frame;
			{
				std::lock_guard<std::mutex> lock(queueMutex);
				if (cap.read(frame)) {
					frameQueue.push(frame);
				}
				else {
					cap.set(cv::CAP_PROP_POS_FRAMES, 0);
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10)); // to redice the CPU compsumtion
		}
	}
}

using namespace cvedia;

int main(int argc, char* argv[]) {
	std::string filename;
	try {
		filename = user_input::program_input(argc, argv);
		const std::vector<std::string> yoloFiles = {
		"model/yolov3.cfg",
		"model/yolov3.weights"
		};

		for (const std::string& file : yoloFiles) {
			if (!std::filesystem::exists(file)) {
				std::cerr << "Error: File not found: " << file << std::endl;
				return EXIT_FAILURE;
			}
		}

		cv::VideoCapture cap;
		cap.open(filename);
		
		if (!cap.isOpened()) {
			std::cerr << "Error: Could not open video file " << filename << std::endl;
			return EXIT_FAILURE;
		}

		std::thread readerThread(reading_video::readFrames, std::ref(cap));

		const std::string modelClasses{ "model/coco_classes.txt" };
		const std::string modelColors{ "model/coco_colors.txt" };

		std::unique_ptr<GuiEngine> guiEngine = std::make_unique<OpenCVGuiEngine>();

		// passing the 2 parameters. The folder model/ SHOULD be at the same level of the code (e.g. folder src/ and include/)
		if (!guiEngine->init(2, modelClasses, modelColors)) {
			std::cerr << "Failed opening config files" << std::endl;
			return EXIT_FAILURE;
		}

		// the model and weights
		std::unique_ptr<InferenceEngine> inferenceEngine = std::make_unique<OpenCVInferenceEngine>();
		inferenceEngine->loadModel(yoloFiles[0], yoloFiles[1]);

		cv::Mat frame;
		while (reading_video::keepRunning) {
			{
				std::lock_guard<std::mutex> lock(reading_video::queueMutex);
				if (!reading_video::frameQueue.empty()) {
					frame = reading_video::frameQueue.front(); // get the oldest frame in queue
					reading_video::frameQueue.pop(); // remove the frame
				}
			}

			if (frame.empty()) {
				break; // TODO: check if this is the better approach
			}
			else {
				const core::ObjectDetectionResult& detectedObjectsInfo = inferenceEngine->inference(frame);
				guiEngine->display(frame.clone(), std::ref(frame), detectedObjectsInfo);
				reading_video::keepRunning = guiEngine->update();
			}
		}

		readerThread.join();
		cap.release();
		guiEngine->release();
	}
	catch (const std::exception& x) {
		std::cerr << "error: " << x.what() << '\n';
		return EXIT_FAILURE;
	}
	return 0;
}
