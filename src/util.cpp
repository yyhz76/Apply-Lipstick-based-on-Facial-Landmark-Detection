#include "util.h"

void parseCommandLineArgs(int argc, char* argv[], CommandLineArgs& args) {
	const std::string keys =
		"{help usage h ?     |      | print this message}"
		"{@model_path        |      | landmark detector path}"
		"{@image_path        |      | input image path}"
		"{@downsample_ratio  |1.0   | image downsample ratio}"
		"{renderFace r       |      | display facial landmarks on the face}"
		;

	CommandLineParser parser(argc, argv, keys);
	parser.about("Virtual Makeup v1.0.0");
	if (argc == 1 || parser.has("help"))
	{
		parser.printMessage();
		exit(EXIT_SUCCESS);
	}

	args.isRenderFace = parser.has("renderFace");
	args.modelPath = parser.get<std::string>(0);
	args.dataPath = parser.get<std::string>(1);
	args.downsampleRatio = parser.get<double>(2);

	if (!parser.check())
	{
		parser.printErrors();
		exit(EXIT_FAILURE);
	}
}

dlib::shape_predictor loadLandmarkDetector(const std::string& model_path) {
	dlib::shape_predictor landmarkDetector;
	try {
		dlib::deserialize(model_path) >> landmarkDetector;
	}
	catch (dlib::serialization_error& e) {
		std::cout << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	return landmarkDetector;
}

// display landmarks on the face
void renderFace(cv::Mat& img, const std::vector<cv::Point2f>& points, cv::Scalar color, int radius) {
	for (int i = 0; i < points.size(); i++) {
		cv::circle(img, points[i], radius, color, -1);
	}

	const int outerLip[] = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
	const int innerLip[] = { 60, 61, 62, 63, 64, 65, 66, 67 };
	std::vector<int> outerLipIndex(outerLip, outerLip + sizeof(outerLip) / sizeof(outerLip[0]));
	std::vector<int> innerLipIndex(innerLip, innerLip + sizeof(innerLip) / sizeof(innerLip[0]));

	for (int i = 0; i < outerLipIndex.size(); i++) {
		cv::circle(img, points[outerLipIndex[i]], radius, Scalar(0, 0, 255), -1);
	}

	for (int i = 0; i < innerLipIndex.size(); i++) {
		cv::circle(img, points[innerLipIndex[i]], radius, Scalar(255, 0, 0), -1);
	}
}

// blend foreground with background using alpha mask: alpha * foreground + (1 - alpha) * background
void alphaBlend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage) {
	Mat fore, back;
	cv::multiply(alpha, foreground, fore, 1 / 255.0);
	cv::multiply(Scalar::all(255) - alpha, background, back, 1 / 255.0);
	add(fore, back, outImage);
}

