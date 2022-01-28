#include "virtual_makeup.h"

int main(int argc, char** argv) {
	
	// parse command line arguments
	CommandLineArgs args;
	parseCommandLineArgs(argc, argv, args);

	Virtual_Makeup app;

	// load face detector
	dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

	// load landmark detector
	dlib::shape_predictor landmarkDetector = loadLandmarkDetector(args.modelPath);

	// read and resize image
	app.readImage(args.dataPath);
	app.resizeImage(args.downsampleRatio);

	// get facial landmarks
	app.getLandmarks(faceDetector, landmarkDetector);

	// render face with landmarks
	if (args.isRenderFace) {
		app.renderFaceForVM();
	}

	app.applyLipstick();

	return 0;
}