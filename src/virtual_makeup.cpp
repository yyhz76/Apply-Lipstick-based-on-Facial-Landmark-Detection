#include "virtual_makeup.h"

Virtual_Makeup::Virtual_Makeup() : 
	_resultWindow(""), 
	_landmarks(std::vector<Point2f>()), 
	_img(Mat()), _imgOut(Mat()), 
	_bgMask(Mat()) 
{}

void Virtual_Makeup::readImage(const std::string& data_path) {
	_img = imread(data_path);
	if (!_img.data)
	{
		std::cout << "Could not open or find the image" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void Virtual_Makeup::resizeImage(double ratio) {
	resize(_img, _img, cv::Size(), 1.0 / ratio, 1.0 / ratio);
}

// display facial landmarks for the virtual makeup image
void Virtual_Makeup::renderFaceForVM() const {
	Mat img_landmarks = _img.clone();
	renderFace(img_landmarks, _landmarks, Scalar(255, 255, 255));
	imshow("face with landmarks", img_landmarks);
}

// get the lip points from landmarks
void Virtual_Makeup::getLipPoints(std::vector<Point>& outerLipPoints, std::vector<Point>& innerLipPoints) const {
	const int outerLip[] = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
	const int innerLip[] = { 60, 61, 62, 63, 64, 65, 66, 67 };

	std::vector<int> outerLipIndex(outerLip, outerLip + sizeof(outerLip) / sizeof(outerLip[0]));
	std::vector<int> innerLipIndex(innerLip, innerLip + sizeof(innerLip) / sizeof(innerLip[0]));

	for (int ind : outerLipIndex) {
		outerLipPoints.push_back(Point(_landmarks[ind].x, _landmarks[ind].y));
	}
	for (int ind : innerLipIndex) {
		innerLipPoints.push_back(Point(_landmarks[ind].x, _landmarks[ind].y));
	}
}

// heuristic to compute Gaussian blur kernel size based on the lip size
int Virtual_Makeup::getBlurKernelSize() const {
	float xdiff = _landmarks[48].x - _landmarks[54].x;
	float ydiff = _landmarks[48].y - _landmarks[54].y;
	float dist = sqrt(xdiff * xdiff + ydiff * ydiff);
	int kernel_size = static_cast<int>(dist) / 10 / 2 * 2 + 1;

	std::cout << "Distance between left lip corner and right lip corner = " << dist << std::endl;
	std::cout << "Gaussian blur kernel size = " << kernel_size << std::endl << std::endl;

	return kernel_size;
}

// return a mask such that only the lip region is left UNMASKED
Mat Virtual_Makeup::getLipMask(Size size, std::vector<Point>& outerLipPoints, std::vector<Point>& innerLipPoints) const {
	Mat mask = Mat::zeros(size.height, size.width, CV_8UC3);
	std::vector<std::vector<Point>> outerLipPointsArray, innerLipPointsArray;
	
	outerLipPointsArray.push_back(outerLipPoints);
	innerLipPointsArray.push_back(innerLipPoints);
	fillPoly(mask, outerLipPointsArray, Scalar(255, 255, 255));
	fillPoly(mask, innerLipPointsArray, Scalar(0, 0, 0));

	return mask;
}

// adjust the color in the region bounded by 'points' and masked by '_bgMask'
void Virtual_Makeup::adjustColor(std::vector<Point>& points) {
	std::cout << std::endl << "Please adjust the colors ... Once finished, press Esc to exit..." << std::endl;

	// create BGR trackbars
	createTrackbar("B", _resultWindow, 0, 255);
	createTrackbar("G", _resultWindow, 0, 255);
	createTrackbar("R", _resultWindow, 0, 255);

	Mat fg = Mat::zeros(_img.size(), _img.type());

	while (1) {
		int b = getTrackbarPos("B", _resultWindow);
		int g = getTrackbarPos("G", _resultWindow);
		int r = getTrackbarPos("R", _resultWindow);

		// create a foreground image with the lip color
		Scalar lipColor(b, g, r);
		fg.setTo(lipColor);

		// blend the foreground lip color image with the original image using the lip region mask
		alphaBlend(_bgMask, fg, _img, _imgOut);
		imshow("before seamless cloning", _imgOut);

		// seamless cloning
		Rect r1 = boundingRect(points);
		Point center = (r1.tl() + r1.br()) / 2;

		seamlessClone(_img, _imgOut, _bgMask, center, _imgOut, NORMAL_CLONE);
		imshow("lip region", _bgMask);
		imshow(_resultWindow, _imgOut);

		// Press ESC to exit
		int c = waitKey(20);
		if (static_cast<char>(c) == 27)
			break;
	}
}

void Virtual_Makeup::applyLipstick() {
	std::cout << "Applying lipstick ..." << std::endl << std::endl;
	
	imshow("original image", _img);

	_resultWindow = "Applying lipstick";
	namedWindow(_resultWindow);

	// output image
	_imgOut = _img.clone();

	// specify outer/inner lip points
	std::vector<Point> outerLipPoints, innerLipPoints;
	getLipPoints(outerLipPoints, innerLipPoints);

	// heuristic to compute Gaussian blur kernel size based on the size of the lip
	int kernel_size = getBlurKernelSize();

	// get lip mask (only the lip region is left UNMASKED)
	_bgMask = getLipMask(_imgOut.size(), outerLipPoints, innerLipPoints);

	// apply gaussian blur so that the color transition around lip boundaries look more natural
	GaussianBlur(_bgMask, _bgMask, Size(kernel_size, kernel_size), 0, 0);

	adjustColor(outerLipPoints);
}

// converts dlib landmarks into a std::vector of 2D points
void Virtual_Makeup::dlibLandmarksToPoints(dlib::full_object_detection& landmarks, std::vector<Point2f>& points) const {
	// Loop over all landmark points
	for (int i = 0; i < landmarks.num_parts(); i++)
	{
		Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
		points.push_back(pt);
	}
}

// get facial landmarks from a face image
void Virtual_Makeup::getLandmarks(dlib::frontal_face_detector& faceDetector, dlib::shape_predictor& landmarkDetector, float FACE_DOWNSAMPLE_RATIO) {
	Mat imgSmall;
	resize(_img, imgSmall, Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);

	// Convert OpenCV image format to Dlib's image format
	dlib::cv_image<dlib::bgr_pixel> dlibIm(_img);
	dlib::cv_image<dlib::bgr_pixel> dlibImSmall(imgSmall);

	// Detect faces in the image
	std::vector<dlib::rectangle> faceRects = faceDetector(dlibImSmall);

	if (faceRects.size() > 0)
	{
		// Pick the biggest face
		dlib::rectangle rect = *std::max_element(faceRects.begin(), faceRects.end(), rectAreaComparator);

		dlib::rectangle scaledRect(
			(long)(rect.left() * FACE_DOWNSAMPLE_RATIO),
			(long)(rect.top() * FACE_DOWNSAMPLE_RATIO),
			(long)(rect.right() * FACE_DOWNSAMPLE_RATIO),
			(long)(rect.bottom() * FACE_DOWNSAMPLE_RATIO)
		);

		dlib::full_object_detection landmarks = landmarkDetector(dlibIm, scaledRect);
		dlibLandmarksToPoints(landmarks, _landmarks);
	}
}

// compare dlib rectangle
bool Virtual_Makeup::rectAreaComparator(dlib::rectangle& r1, dlib::rectangle& r2) {
	return r1.area() < r2.area();
}