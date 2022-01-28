#pragma once

#include "util.h"

class Virtual_Makeup {
public:
	Virtual_Makeup();
	void readImage(const std::string& data_path);
	void resizeImage(double ratio);
	void renderFaceForVM() const;
	void applyLipstick();
	void getLandmarks(dlib::frontal_face_detector& faceDetector, dlib::shape_predictor& landmarkDetector, float FACE_DOWNSAMPLE_RATIO = 1);

private:
	void getLipPoints(std::vector<Point>& outerLipPoints, std::vector<Point>& innerLipPoints) const;
	void adjustColor(std::vector<Point>& colorRegion);
	void dlibLandmarksToPoints(dlib::full_object_detection& landmarks, std::vector<Point2f>& points) const;
	int getBlurKernelSize() const;
	Mat getLipMask(Size size, std::vector<Point>& outerLipPoints, std::vector<Point>& innerLipPoints) const;
	static bool rectAreaComparator(dlib::rectangle& r1, dlib::rectangle& r2);

private:
	std::string _resultWindow;
	std::vector<Point2f> _landmarks;
	Mat _img;
	Mat _imgOut;
	Mat _bgMask;
};