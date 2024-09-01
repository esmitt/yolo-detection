#include <iostream>
#include <string_view>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <queue>
#include <iostream>
#include <opencv2/dnn.hpp>

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

std::queue<cv::Mat> frameQueue;  // Queue to store frames
std::mutex queueMutex;           // Mutex for thread-safe access to the queue
bool keepRunning = true;         // Flag to control the reading and displaying

void readFrames(cv::VideoCapture& cap) {
	while (keepRunning) {
		cv::Mat frame;
		{
			std::lock_guard<std::mutex> lock(queueMutex); // Lock mutex for thread-safe access
			if (cap.read(frame)) {                        // Read frame from video
				frameQueue.push(frame);                   // Push frame to the queue
			}
			else {
					cap.set(cv::CAP_PROP_POS_FRAMES, 0);
					//keepRunning = true;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Sleep to reduce CPU usage
	}
}

std::vector<std::string> loadClassesNames(const std::string& filename) {
	std::ifstream ifs{ filename };
	if (!ifs) {
		throw std::runtime_error("Failed to open file: " + filename);
	}
	std::vector<std::string> classNames;
	std::string line;
	while (std::getline(ifs, line)) {
		classNames.emplace_back(std::move(line));
	}

	return classNames;
}

std::vector<cv::Scalar> loadColors(const std::string& filename) {
	std::ifstream ifs{ filename };
	if (!ifs) {
		throw std::runtime_error("Failed to open file: " + filename);
	}
	std::vector<cv::Scalar> colors;
	std::string line;
	while (std::getline(ifs, line)) {
		line.erase(std::remove(line.begin(), line.end(), ','), line.end());
		std::istringstream iss( line );
		
		int r, g, b;
		if (iss >> r >> g >> b) {
			colors.push_back(cv::Scalar(r, g, b));
		}
		else{
			std::cerr << "Error parsing line: " << line << std::endl;
		}
	}
	return colors;
}

int main(int argc, char* argv[]) {
	std::string filename;
	try {
		filename = "C:\\code\\cvedia\\data\\test.mp4";// { user_input::program_input(argc, argv) };
	}
	catch (const std::exception& x) {
		std::cerr << "dog: " << x.what() << '\n';
		std::cerr << "usage: dog [-n|--number] [-E|--show-ends] <input_file> ...\n";
		return EXIT_FAILURE;
	}

	cv::VideoCapture cap(filename);
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open video file " << filename << std::endl;
		return EXIT_FAILURE;
	}
	
	std::thread readerThread(readFrames, std::ref(cap));

	// network
	const std::string modelConfig{"C:\\code\\Cvedia-Esmitt\\model\\yolov3.cfg"};
	const std::string modelWeights{ "C:\\code\\Cvedia-Esmitt\\model\\yolov3.weights" };
	const std::string modelClasses{ "C:\\code\\Cvedia-Esmitt\\model\\coco_classes.txt" };
	const std::string modelColors{ "C:\\code\\Cvedia-Esmitt\\model\\coco_colors.txt" };
	
	std::vector<cv::Scalar> classesColors;
	std::vector<std::string> classesNames;
	try {
		if (!std::filesystem::exists(modelConfig)) {
			throw std::runtime_error(modelConfig + ": No such file");
		}
		if (!std::filesystem::exists(modelWeights)) {
			throw std::runtime_error(modelWeights + ": No such file");
		}
		classesNames = loadClassesNames(modelClasses);
		classesColors = loadColors(modelColors);
	}
	catch (const std::exception& x) {
		std::cerr << "dog: " << x.what() << '\n';
		return EXIT_FAILURE;
	}

	//https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
	cv::dnn::Net network = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
	network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	
	network.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
	
	// determine output layers
	std::vector<int> indexesOfOutputLayers{ network.getUnconnectedOutLayers() };
	std::vector<std::string> layerNames{network.getLayerNames()};
	std::vector<std::string> namesOfOutputLayers;
	for (size_t i = 0; i < indexesOfOutputLayers.size(); i++) {
		const int index{ indexesOfOutputLayers[i] - 1 };
		//TODO: check boundaries of index
		namesOfOutputLayers.emplace_back(layerNames[index]);
	}

	if (network.empty()) {
		std::cerr << "Could not load the network!" << std::endl;
		return -1;
	}

	constexpr double scaleFactor {1.0/255.0 };
	const cv::Size sizeInput(416, 416);
	const cv::Scalar zeroScale(0);

	while (keepRunning) {
		cv::Mat frame;
		{
			std::lock_guard<std::mutex> lock(queueMutex); // Lock mutex for thread-safe access
			if (!frameQueue.empty()) {
				frame = frameQueue.front();  // Get the oldest frame in the queue
				frameQueue.pop();            // Remove the frame from the queue
			}
		}

		if (!frame.empty()) {
			cv::Mat original(frame.clone());

			// convert to blob
			cv::Mat inputBlob{ cv::dnn::blobFromImage(frame, scaleFactor, sizeInput, zeroScale, true, false) };
			network.setInput(inputBlob);
			std::vector<cv::Mat> outputs;
			network.forward(outputs, namesOfOutputLayers);

			std::vector<cv::Rect> boxes;
			std::vector<int> classIds;
			std::vector<float> confidences;

			for (const auto& output : outputs) {
				for (int index = 0; index < output.rows; index++) {
					cv::Mat row{ output.row(index) };
					cv::Mat scores{ row.colRange(5, output.cols) };

					// Find the class with the highest confidence
					double maxClassConfidence;
					cv::Point classIdPoint;
					cv::minMaxLoc(scores, nullptr, &maxClassConfidence, nullptr, &classIdPoint);

					const float objectness = row.at<float>(4);
					if (objectness > 0.5f) {
						int centerX = static_cast<int>(row.at<float>(0) * frame.cols);
						int centerY = static_cast<int>(row.at<float>(1) * frame.rows);
						int width = static_cast<int>(row.at<float>(2) * frame.cols);
						int height = static_cast<int>(row.at<float>(3) * frame.rows);
						int left = centerX - width * 0.5f;
						int top = centerY - height * 0.5f;

						boxes.emplace_back(cv::Rect(left, top, width, height));
						classIds.push_back(classIdPoint.x); // The class index with the highest score
						confidences.push_back(static_cast<float>(maxClassConfidence));
					}
				}
			}

			// Apply Non-Maximum Suppression (NMS)
			std::vector<int> indices;
			float nmsThreshold = 0.4f; // NMS threshold. Adjust this value based on your use case
			cv::dnn::NMSBoxes(boxes, confidences, 0.5f, nmsThreshold, indices);

			//// Draw bounding boxes with labels and confidences (no NMS applied)
			//for (size_t i = 0; i < boxes.size(); ++i) {
			//	cv::Rect box = boxes[i];
			//	int classId = classIds[i];
			//	cv::Scalar color = classesColors[classId];  // Get the color for the detected class
			//	rectangle(frame, box, color, 2);  // Draw the bounding box with the corresponding class color

			//	std::string label = cv::format("%.2f", confidences[i]);
			//	if (!classesColors.empty()) {
			//		//CV_Assert(classId < static_cast<int>(classNames.size()));
			//		label = classesNames[classId] + ": " + label;  // Append the class name to the label
			//	}
			//	int baseLine;
			//	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			//	int top = std::max(box.y, labelSize.height);
			//	rectangle(frame, cv::Point(box.x, top - round(1.5 * labelSize.height)),
			//		cv::Point(box.x + round(1.5 * labelSize.width), top + baseLine),
			//		cv::Scalar(255, 255, 255), cv::FILLED);
			//	putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
			//}

			for (size_t i = 0; i < indices.size(); ++i) {
				int idx = indices[i];
				cv::Rect box = boxes[idx];
				int classId = classIds[idx];
				cv::Scalar color = classesColors[classId];  // Get the color for the detected class
				rectangle(frame, box, color, 2);  // Draw the bounding box with the corresponding class color

				std::string label = cv::format("%.2f", confidences[idx]);
				if (!classesNames.empty()) {
					// Ensure that the class ID is within the bounds of the class names vector
					CV_Assert(classId < static_cast<int>(classesNames.size()));
					label = classesNames[classId] + ": " + label;  // Append the class name to the label
				}
				int baseLine;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				int top = std::max(box.y, labelSize.height);
				rectangle(frame, cv::Point(box.x, top - round(1.5 * labelSize.height)),
					cv::Point(box.x + round(1.5 * labelSize.width), top + baseLine),
					cv::Scalar(255, 255, 255), cv::FILLED);
				putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
			}

			cv::Mat combined;
			cv::hconcat(original, frame, combined);
			cv::imshow("CVEDIA C++ Tech Test", combined);
			if (cv::waitKey(1) == 'q') {
				keepRunning = false;
			}
			
		}
	}

	// Clean up
	readerThread.join();  // Wait for the reader thread to finish
	cap.release();        // Release the video capture object
	cv::destroyAllWindows(); // Close all OpenCV windows

	return EXIT_SUCCESS;
}