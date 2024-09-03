# Object Detection with OpenCV DNN

This project implements an object detection application using OpenCV's Deep Neural Network (DNN) module. The application reads video frames, performs object detection using a YOLO model, and displays the results in a GUI window.

## Prerequisites

- **OpenCV**: Ensure OpenCV is installed with contrib modules. If `find_package(OpenCV REQUIRED)` does not locate OpenCV, specify the path in the `CMakeLists.txt` file.

Ensure that the `model/` directory contains `coco_classes.txt`, `coco_colors.txt`, `yolov3.cfg`, and `yolov3.weights` at the same level as the `src/` and `include/` directories.

## Inference Engine Configuration

The inference engine is set with the following lines in `opencvinferenceengine.cpp`:

```cpp
m_network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
m_network.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
```

You can modify these settings as needed to change the backend or target for inference.

### Notes

This code has not been tested on a Linux-based machine due to the lack of availability. However, it has been successfully built using GitHub Actions with a Linux-based configuration.