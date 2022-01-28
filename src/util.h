#pragma once

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

using namespace cv;

struct CommandLineArgs {
	bool isRenderFace;
	double downsampleRatio;
	std::string dataPath;
	std::string modelPath;
};

void parseCommandLineArgs(int argc, char* argv[], CommandLineArgs& args);
void alphaBlend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage);
void renderFace(Mat& img, const std::vector<Point2f>& points, Scalar color, int radius = 3);
dlib::shape_predictor loadLandmarkDetector(const std::string& model_path);

