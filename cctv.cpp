//#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2\videoio.hpp"
#include <concurrent_queue.h>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "qcvcamshifttracker.h"
using namespace cv;
using namespace std;
using namespace concurrency;

// Thread
class CameraStreamer {
public:
	//this holds camera stream urls
	vector<string> camera_source;
	//this holds usb camera indices
	vector<int> camera_index;
	//this holds OpenCV VideoCapture pointers
	vector<VideoCapture*> camera_capture;
	//this holds queue(s) which hold images from each camera
	vector<concurrent_queue<Mat>*> frame_queue;
	//this holds thread(s) which run the camera capture process
	vector<thread*> camera_thread;

	//Constructor for IP Camera capture
	CameraStreamer(vector<string> source);
	//Constructor for USB Camera capture
	CameraStreamer(vector<int> index);
	//Destructor for releasing resource(s)
	~CameraStreamer();

private:
	bool isUSBCamera;
	int camera_count;
	//initialize and start the camera capturing process(es)
	void startMultiCapture();
	//release all camera capture resource(s)
	void stopMultiCapture();
	//main camera capturing process which will be done by the thread(s)
	void captureFrame(int index);
};

CameraStreamer::CameraStreamer(vector<string> stream_source)
{
	camera_source = stream_source;
	camera_count = camera_source.size();
	isUSBCamera = false;

	startMultiCapture();
}

CameraStreamer::CameraStreamer(vector<int> capture_index)
{
	camera_index = capture_index;
	camera_count = capture_index.size();
	isUSBCamera = true;

	startMultiCapture();
}

CameraStreamer::~CameraStreamer()
{
	stopMultiCapture();
}

void CameraStreamer::captureFrame(int index)
{
	VideoCapture *capture = camera_capture[index];
	while (true)
	{
		Mat frame;
		//Grab frame from camera capture
		(*capture) >> frame;
		//Put frame to the queue
		if (frame_queue[index]->empty())
			frame_queue[index]->push(frame);
		//relase frame resource
		frame.release();
	}
}

void CameraStreamer::startMultiCapture()
{
	VideoCapture *capture;
	thread *t;
	concurrent_queue<Mat> *q;
	for (int i = 0; i < camera_count; i++)
	{
		//Make VideoCapture instance
		if (!isUSBCamera) {
			string url = camera_source[i];
			capture = new VideoCapture(url);
			cout << "Camera Setup: " << url << endl;
		}
		else {
			int idx = camera_index[i];
			capture = new VideoCapture(idx);
			cout << "Camera Setup: " << to_string(idx) << endl;
		}

		//Put VideoCapture to the vector
		camera_capture.push_back(capture);

		//Make thread instance
		t = new thread(&CameraStreamer::captureFrame, this, i);

		//Put thread to the vector
		camera_thread.push_back(t);

		//Make a queue instance
		q = new concurrent_queue<Mat>;

		//Put queue to the vector
		frame_queue.push_back(q);
	}
}

void CameraStreamer::stopMultiCapture()
{
	VideoCapture *cap;
	for (int i = 0; i < camera_count; i++) {
		cap = camera_capture[i];
		if (cap->isOpened()) {
			//Relase VideoCapture resource
			cap->release();
			cout << "Capture " << i << " released" << endl;
		}
	}
}

// draw arrow
void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
	cv::Scalar& color, int thickness, int lineType)
{
	const double PI = 3.1415926;
	Point arrow;
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
}



int main(int argc, char *argv[]) {
	// FPS
	time_t start, end;
	char strBuffer[64] = { 0, };
	
	// camera connect
	vector<string> capture_source = {
		"rtsp://admin:git7100@@203.249.22.100:554/profile2/media.smp",
		"rtsp://admin:git7100@@203.249.22.20:554/profile2/media.smp"
	};
	vector<int> capture_index = { 0, 1 };
	vector<string> label;
	for (int i = 0; i < capture_source.size(); i++) {
		string title = "CCTV " + to_string(i);
		label.push_back(title);
	}
	CameraStreamer cam(capture_source);

	// image : original image, grayscale image
	Mat limage, rimage;
	Mat lgray, rgray;

	// image ROI
	Mat beforeLImage, beforeRImage;
	Rect rect(0, 250, 800 - 0, 450 - 250);

	// MOG2 : Background Substraction
	Mat mog, mog_depth;
	Ptr <BackgroundSubtractorMOG2> pMOG2;
	//Ptr <BackgroundSubtractorMOG2> pMOG2_depth;
	pMOG2 = createBackgroundSubtractorMOG2(600, 16, true);
	//pMOG2_depth = createBackgroundSubtractorMOG2();

	// Mophology
	Mat element(5, 5, CV_8U, Scalar(1));

	// connected ComponentsWithstats
	Mat binary, color;
	Mat labels, stats, centroids;

	// optical flow
	UMat  flowUmat, prevgray;
	Mat flow, img, original;
	Mat original1;

	// StereoSGBM parameter
	//int ndisparities = 16 * 6;   // Range of disparity, 윈도우 사이즈 클수록 크게 지정
	//int SADWindowSize = 11; // Size of the block window. Must be odd, 윈도우 사이즈 클수록 크게 지정
	int ndisparities = 16 * 5;   // Range of disparity, 윈도우 사이즈 클수록 크게 지정
	int SADWindowSize = 7; // Size of the block window. Must be odd, 윈도우 사이즈 클수록 크게 지정
						   // StereoSGBM create
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, ndisparities, SADWindowSize);
	int P1 = 200, P2 = 2000;
	//sgbm->setPreFilterCap(40);      // disparity 부드러움 제어
	//sgbm->setMinDisparity(-9);      // 최소 disparity, 적절히 조절
	//sgbm->setUniquenessRatio(7);   // 두번째로 가장 좋은 값 획득?, 5-10
	//sgbm->setSpeckleWindowSize(150);   // 노이즈 제거를 위한 최대 크기, 50-200
	//sgbm->setSpeckleRange(12);      // 연결된 각 픽셀 내 disparity 변화, 1-2
	//sgbm->setDisp12MaxDiff(10);
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	//sgbm->setPreFilterCap(32);
	//sgbm->setUniquenessRatio(0);

	// wls flitering
	//double lambda = 500.0;  // 객체 구분성? 100 정도면 경계가 흐릿함 
	//double sigma = 4.5; //높일수록 프레임 빨라짐 

	// Object Tracking
	vector<Rect> prevRect;
	Mat bp;
						/*
						cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, 5.5);

						resize = 0.7 기준, 프레임 5 정도
						int ndisparities = 16 * 3;
						int SADWindowSize = 13;
						double lambda = 500.0;
						double sigma = 3.5;

						resize = 0.6 기준, 프레임 6-7 정도
						int ndisparities = 16 * 2;
						int SADWindowSize = 9;
						double lambda = 500.0;
						double sigma = 4.5;

						resize = 0.5 기준, 프레임 7.8 정도
						int ndisparities = 16 * 2;
						int SADWindowSize = 9;
						double lambda = 400.0;
						double sigma = 5.5;

						resize = 0.3 기준, 프레임 20 정도
						int ndisparities = 16 * 1;
						int SADWindowSize = 7;
						double lambda = 100.0;
						double sigma = 1.5;
						*/


	// 동영상 저장

	//Size size = Size((int)cam.camera_capture[0]->get(CAP_PROP_FRAME_WIDTH),
	//	(int)cam.camera_capture[0]->get(CAP_PROP_FRAME_HEIGHT));
		//Size(800, 450);

	//웹캠에서 캡처되는 속도를 가져옴
	//int fps = cam.camera_capture[1]->get(CAP_PROP_FPS);

	// 초기화
	VideoWriter leftWriter, rightWriter, depthWriter;
	int codec = CV_FOURCC('P', 'I', 'M', '1');  // select desired codec (must be available at runtime)


	double fps = cam.camera_capture[0]->get(CAP_PROP_FPS); // framerate of the created video stream
	//cout << "ddd" << fps << endl;
	
	int videonum = 1;
	
	cout << "파일명을 입력하세요 (숫자)" << endl;
	cin >> videonum;

	string leftFilename = "Left (" + to_string(videonum) + ").avi";             // name of the output video file
	string rightFilename = "Right (" + to_string(videonum) + ").avi";             // name of the output video file
	//string depthFilename = "Depth (" + to_string(videonum) + ").avi";

	leftWriter.open(leftFilename, codec, fps, Size(800,450-250) , true);
	rightWriter.open(rightFilename, codec, fps, Size(800, 450 - 250), true);
	//depthWriter.open(depthFilename, codec, fps, Size(800*0.8, (450-250)*0.8), false);

	// check if we succeeded
	if (!leftWriter.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return -1;
	}
	if (!rightWriter.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return -1;
	}
	//if (!depthWriter.isOpened()) {
	//	cerr << "Could not open the output video file for write\n";
	//	return -1;
	//}

	

	// FPS start
	time(&start);
	int counter = 0;
	int trackCounter1 = 0;
	int trackCounter2 = 0;

	vector <QcvCAMshiftTracker> camShiftTrackers1;
	vector <QcvCAMshiftTracker> camShiftTrackers2;

	// rotate
	int iAngle = 0;
	Mat matRotatedFrame;

	// fps 조절
	int fpsi = 0;

	while (waitKey(1) != 27) {
		trackCounter1++;
		trackCounter2++; 


		Mat temp;
		// 화면에 예쁘게 띄우기 위한 변수
		int w = 800 * 0.8, h = (450 - 250)*0.8 + 40;

		if (cam.frame_queue[0]->try_pop(beforeLImage)) {
			// Image ROI sub

			//사이즈 수정중, if 쪽에 beforeLImage 넣고, limage=beforeLiamge(rect) 하면됌
			// 		if (cam.frame_queue[0]->try_pop(beforeLImage)) {
			// 원본이미지 쓰려면 try_pop(limage)

			limage = beforeLImage(rect);
			// 동영상
		
			leftWriter.write(limage);

			// resize
			resize(limage, limage, Size(), 0.8, 0.8); // 원래 0.6 (CCTV 수리 후 2017.08.16)
			namedWindow("left video");
			moveWindow("left video", 0 * w, 0 * h);
			imshow("left video", limage);

			// CAMSHIFT
			original = limage.clone();
			original1 = limage.clone();

			cvtColor(limage, lgray, CV_BGR2GRAY);
		}
		//outputVideo << beforeLImage;
		if (cam.frame_queue[1]->try_pop(beforeRImage)) {
			// Image ROI sub
			//사이즈 수정중
			rimage = beforeRImage(rect);


			// rotate
			//Mat matRotation = getRotationMatrix2D(Point(rimage.cols / 2, rimage.rows / 2), (iAngle - 1), 1);
			
			//warpAffine(rimage, matRotatedFrame, matRotation, rimage.size());
			//resize(matRotatedFrame, matRotatedFrame, Size(), 0.6, 0.6);
			//imshow("rotate", matRotatedFrame);
			//moveWindow("rotate", 2 * w, 0 * h);

			



			// 동영상
			rightWriter.write(rimage);
			// resize
			resize(rimage, rimage, Size(), 0.8, 0.8);
			namedWindow("right video");
			moveWindow("right video", 1 * w, 0 * h);
			imshow("right video", rimage);

			// 오른쪽 동영상 기울임 처리 rimage -> matRotatedFrame (1)
			cvtColor(rimage, rgray, CV_BGR2GRAY);
		}

		// MOG : Background Substract

		// mog 매개변수 true 일 때 
		GaussianBlur(lgray, lgray, Size(7, 7), 1.5, 1.5);
		pMOG2->apply(lgray, mog);
		namedWindow("Original MOG");
		moveWindow("Original MOG", 0 * w, 2 * h);
		imshow("Original MOG", mog);

		
		dilate(mog, mog, Mat(), Point(-1, -1), 2);
		erode(mog, mog, Mat(), Point(-1, -1), 1);
		dilate(mog, mog, Mat(), Point(-1, -1), 1);
		threshold(mog, mog, 200, 255, THRESH_BINARY);
		//dilate(mog, mog, Mat(), Point(-1, -1), 1);

		// mog 매개변수 false 일 때 
		//pMOG2->apply(lgray, mog);
		//namedWindow("Original MOG");
		//moveWindow("Original MOG", 3 * w, 0 * h);
		//imshow("Original MOG", mog);
		//erode(mog, mog, Mat(), Point(-1, -1), 1);
		//dilate(mog, mog, Mat(), Point(-1, -1), 2);

		///////////////////////////////////////

		namedWindow("MOG");
		moveWindow("MOG", 1 * w, 2 * h);
		imshow("MOG", mog);

		// object labeling : MOG로 객체 추출 한 것을 라벨링 하는 과정
		threshold(mog, binary, 127, 255, THRESH_BINARY);

		cvtColor(binary, color, COLOR_GRAY2BGR);

		int numOfLables = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
		int n = 0; // Group number



		//string title = "CCTV " + to_string(i);

		// labeling 
		for (int y = 0; y < labels.rows; ++y) {
			int *label = labels.ptr<int>(y);
			Vec3b* pixel = color.ptr<Vec3b>(y);
			for (int x = 0; x < labels.cols; ++x) {
				if (label[x]) {
					pixel[x][2] = 255;
					pixel[x][1] = 222;
					pixel[x][0] = 0;
				}
			}
		}


		// Rectangle Grouping 
		for (int j = 1; j < numOfLables; j++) {
			int area = stats.at<int>(j, CC_STAT_AREA);
			int left = stats.at<int>(j, CC_STAT_LEFT);
			int top = stats.at<int>(j, CC_STAT_TOP);
			int width = stats.at<int>(j, CC_STAT_WIDTH);
			int height = stats.at<int>(j, CC_STAT_HEIGHT);

			if (area >= 100) {
				n++;
				rectangle(color, Point(left, top), Point(left + width, top + height),
					Scalar(0, 0, 255), 2);

				//int x = centroids.at<double>(j, 0); //중심좌표
				//int y = centroids.at<double>(j, 1);
				//circle(color, Point(x, y), 5, Scalar(255, 0, 0), 1);
				string st = "Detect" + to_string(n);;
				//putText(color, to_string(j), Point(left + 20, top + 20),FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
				putText(color, st, Point(left - 0, top - 5), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255), 1.5);


				// CAMShift
				if (trackCounter1 == 1) {
					QcvCAMshiftTracker newTracker1;
					newTracker1.setCurrentRect(Rect(left, top, width, height));
					camShiftTrackers1.push_back(newTracker1);
				}
			}

		}

		namedWindow("object labeling");
		moveWindow("object labeling", 0 * w, 3 * h);
		imshow("object labeling", color);



		QcvCAMshiftTracker::setMainImage(original1);
		for (int trackText = 1, i = 0; i < camShiftTrackers1.size(); i++)
		{
			if (camShiftTrackers1[i].trackCurrentRect().boundingRect().area() > 5)
			{
				//cv::ellipse(original1, camShiftTrackers1[i].trackCurrentRect(), cv::Scalar(0, 255, 0), 2, CV_AA);
				cv::rectangle(original1, camShiftTrackers1[i].trackCurrentRect().boundingRect(), cv::Scalar(0, 0, 255), 2);
				circle(original1, camShiftTrackers1[i].trackCurrentRect().center, 4, Scalar(0, 0, 255), -1);
				String tst = "OBJECT" + to_string(trackText++);
				putText(original1, tst,
					Point(camShiftTrackers1[i].trackCurrentRect().boundingRect().x,
						camShiftTrackers1[i].trackCurrentRect().boundingRect().y - 5),
					FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0), 0.5);

				//line(original1, camShiftTrackers1[i-1].trackCurrentRect().center, camShiftTrackers1[i].trackCurrentRect().center, Scalar(0, 0, 255), 1);
			}
		}
		if (trackCounter1 == 10) {

			camShiftTrackers1.clear();
			trackCounter1 = 0;
		}


		//imshow("RGB based CAMSHIFT Tracking", original1);
		//moveWindow("RGB based CAMSHIFT Tracking", 0 * w, 2 * h);


		// optical flow
		/*
		cvtColor(color, color, COLOR_BGR2GRAY);

		if (prevgray.empty() == false) {
			calcOpticalFlowFarneback(prevgray, mog, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
			flowUmat.copyTo(flow);
			// By y += 5, x += 5 you can specify the grid 
			// draw
			for (int opticalY = 0; opticalY < original.rows; opticalY += 5) {
				for (int opticalX = 0; opticalX < original.cols; opticalX += 5) {
					// get the flow from y, x position * 10 for better visibility
					const Point2f flowatxy = flow.at<Point2f>(opticalY, opticalX) * 10;
					// draw line at flow direction
					if (abs(flowatxy.x + flowatxy.y) > 10 && abs(flowatxy.x + flowatxy.y) < 30) {
						//line(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 0, 0));
						drawArrow(original, Point(opticalX, opticalY), Point(cvRound(opticalX + flowatxy.x), cvRound(opticalY + flowatxy.y)), 10, 15, Scalar(255, 0, 0), 0.5, 4);
					}
					// draw initial point
					//circle(original, Point(x, y), 1, Scalar(0, 0, 0), -1);
					//}

					//if (abs(flowatxy.x + flowatxy.y) > 15 && abs(flowatxy.x + flowatxy.y) < 30)
					//rectangle(original, Rect(x, y, x + flowatxy.x, y + flowatxy.y), Scalar(0, 255, 0), 2);
				}

			}
			namedWindow("Optical Flow");
			moveWindow("Optical Flow", 0 * w, 1 * h);
			imshow("Optical Flow", original);
			mog.copyTo(prevgray);

		}
		else {
			mog.copyTo(prevgray);
		}
		*/

		//StereoSGBM
		//sgbm->compute(lgray, rgray, disp);
		//cv::normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

		//Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
		//Ptr<StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(sgbm);
		//double matching_time = (double)getTickCount();

		Mat left_disp, right_disp;
		//// 오른쪽 동영상 기울임 처리 rimage -> matRotatedFrame (2)
		sgbm->compute(limage, rimage, left_disp);

		Mat imgDisparity8U = Mat(limage.rows, limage.cols, CV_8UC1);
		cv::normalize(left_disp, imgDisparity8U, 0, 255, CV_MINMAX, CV_8U);
		//right_matcher->compute(rimage, limage, right_disp);
		//matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();

		//wls_filter->setLambda(lambda);
		//wls_filter->setSigmaColor(sigma);
		double filtering_time = (double)getTickCount();
		Mat filtered_disp;

		//wls_filter->filter(left_disp, limage, filtered_disp, right_disp);
		filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();

		Mat raw_disp_vis;
		cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, 2);

		// depth 영상 저장
		//depthWriter << raw_disp_vis;

		cv::namedWindow("raw disparity");
		cv::moveWindow("raw disparity", 1 * w, 1 * h);
		cv::imshow("raw disparity", raw_disp_vis);



		//Mat filtered_disp_vis;
		//cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, 5.5);
		//namedWindow("filtered disparity");
		//moveWindow("filtered disparity", 0 * w, 1 * h);
		//imshow("filtered disparity", filtered_disp_vis);

		//Mat depthGaussian;
		//GaussianBlur(filtered_disp_vis, depthGaussian, Size(9, 9), 2, 2);
		//pMOG2->apply(filtered_disp_vis, mog_depth);
		//threshold(mog_depth, mog_depth, 127, 255, CV_THRESH_BINARY);

		//dilate(mog_depth, mog_depth, Mat(), Point(-1, -1), 2);
		//erode(mog_depth, mog_depth, Mat(), Point(-1, -1), 1);
		//imshow("MOG_depth", mog_depth);



		// depth image 위에 color segment 그리기
		// labels : 탐지된 라벨들
		// label : 각 라벨의 좌표 

		/* 수정 전 코드 */
		//Mat color_depth;
		//cvtColor(filtered_disp_vis, color_depth, CV_GRAY2BGR);
		//for (int y = 0; y < labels.rows; ++y) {
		//	int *label = labels.ptr<int>(y);
		//	Vec3b* colorPixel = color_depth.ptr<Vec3b>(y);
		//	for (int x = 0; x < labels.cols; ++x) {
		//		if (label[x]) { // 라벨링된 라벨이 있으면 5
		//			colorPixel[x][2] = 55;
		//			colorPixel[x][1] = 0;
		//			colorPixel[x][0] = 55;
		//		}
		//	}
		//}

		Mat color_depth;
		Mat depthSegment = raw_disp_vis.clone();
		Mat colorSegment;
		cvtColor(raw_disp_vis, color_depth, CV_GRAY2BGR);
		cvtColor(raw_disp_vis, colorSegment, CV_GRAY2BGR);

		typedef vector<Point> labalpoint;
		vector<labalpoint> pointarray;

		vector<int> compareVec;
		//	vector<Vec3b> segmentColor;
		int compareVecnum = -1;
		//int r = -1;

		vector<Rect> segmentPos;

		for (int y = 0; y < labels.rows; ++y) {
			int *label = labels.ptr<int>(y);
		//	Vec3b* colorPixel = color_depth.ptr<Vec3b>(y);
			uchar* depthPixel = depthSegment.ptr<uchar>(y); // depth segment를 위한 변수
			Vec3b* colorSegmentPixel = colorSegment.ptr<Vec3b>(y); // color segment를 위한 변수

			// color segment 결과를 depth map에 뿌려줌 (캡쳐해야함)
			for (int x = 0; x < labels.cols; ++x) {
				if (label[x]) {

					// Color Segment 정보 
					colorSegmentPixel[x][2] = 255;
					colorSegmentPixel[x][1] = 0;
					colorSegmentPixel[x][0] = 255;
				}
			}

			// depth Segment
			// color segment 된 라벨링 된 영역 안에서 depth 값(disparity)을 가져와 Segment
			for (int x = 0; x < labels.cols; ++x) {
				if (label[x]) {
					for (int i = 0; i < compareVec.size(); i++) {
						compareVecnum = -1; 
						// 다른 값이면 

						// depth 값 조절 , depthPixel이 현재값, compareVec가 비교하려는 값(고정)
						if (depthPixel[x] - 5 < compareVec[i] && depthPixel[x] + 5 > compareVec[i]) { // original =3, 2017.09.04 수정
							// 같은색 라벨링 되는 부분 거리에 따라 분리 하는 부분
							if ((segmentPos[i].x - 5 < x) && (segmentPos[i].x + segmentPos[i].width + 5 > x)) {
								if ((segmentPos[i].y - 20 < y && (segmentPos[i].y + segmentPos[i].height + 20 > y))) {
									compareVecnum = i; // 해당 벡터정보를 넣고 if문 빠져나감, pointarray[].push_back
									short int si = 3;
									break;
								}
							}
						}
					}

					if (compareVecnum == -1) { // 다른 라벨이 아무것도 없을 때
						compareVec.push_back(depthPixel[x]);
						compareVecnum = compareVec.size() - 1;
						//Vec3b temp(rand() % 255, rand() % 255, rand() % 255);
						//segmentColor.push_back(temp); // cl 이라는 벡터변수에 컬러값으로 세그먼트된 색을 넣음
						labalpoint temparray;
						pointarray.push_back(temparray);

						//for(int recNum=0; recNum<compareVecnum; recNum++)
						segmentPos.push_back(Rect(x,y,0,0));
						//r = segmentPos.size() - 1;
					}


					//colorPixel[x] = segmentColor[compareVecnum]; // color 값 저장 
					pointarray[compareVecnum].push_back(Point(x, y)); // compareVecnum번째 라벨 저장
					if (segmentPos[compareVecnum].x > x)
						segmentPos[compareVecnum].x = x;
					if (segmentPos[compareVecnum].x + segmentPos[compareVecnum].width < x)
						segmentPos[compareVecnum].width = x - segmentPos[compareVecnum].x;

					if (segmentPos[compareVecnum].y > y)
						segmentPos[compareVecnum].y = y;
					if (segmentPos[compareVecnum].y + segmentPos[compareVecnum].height < y)
						segmentPos[compareVecnum].height = y - segmentPos[compareVecnum].y;
					
				}

			}
		}

		//// Depth based Labeling
		Mat depthLabelImage(color_depth.rows, color_depth.cols, CV_8UC3, cv::Scalar(0)); // Mat 생성 
		//Mat camTrack;
		//camTrack = original.clone();
		
		for (int printtext = 1, labelnum = 0; labelnum < pointarray.size(); labelnum++) {
			int depthLeft=INT_MAX, depthRight=0, depthWidth=0, depthHeight=0, depthTop=INT_MAX, depthBottom=0;

			//Vec3b temp((compareVec[labelnum]&0x7)<<5, ((compareVec[labelnum]>>3) & 0x7) << 5, ((compareVec[labelnum]>>6) & 0x7) << 5);
			Vec3b temp(rand() % 255, rand() % 255, rand() % 255);
			if (pointarray[labelnum].size() > 20) {
				for (int xx = 0; xx < pointarray[labelnum].size(); xx++) {
					//	cout << labelnum << " : " << pointarray[labelnum][xx] << endl;
					depthLabelImage.at<Vec3b>(pointarray[labelnum][xx].y, pointarray[labelnum][xx].x)[0] = temp[0];
					depthLabelImage.at<Vec3b>(pointarray[labelnum][xx].y, pointarray[labelnum][xx].x)[1] = temp[1];
					depthLabelImage.at<Vec3b>(pointarray[labelnum][xx].y, pointarray[labelnum][xx].x)[2] = temp[2];

					color_depth.at<Vec3b>(pointarray[labelnum][xx].y, pointarray[labelnum][xx].x)[0] = temp[0];
					color_depth.at<Vec3b>(pointarray[labelnum][xx].y, pointarray[labelnum][xx].x)[1] = temp[1];
					color_depth.at<Vec3b>(pointarray[labelnum][xx].y, pointarray[labelnum][xx].x)[2] = temp[2];

					//color_depth.ptr<Vec3b>(pointarray[labelnum][xx].x, pointarray[labelnum][xx].y)[0] = temp[0]; // 파란색으로만 나옴 
					//color_depth.ptr<Vec3b>(pointarray[labelnum][xx].x, pointarray[labelnum][xx].y)[1] = temp[1];
					//color_depth.ptr<Vec3b>(pointarray[labelnum][xx].x, pointarray[labelnum][xx].y)[2] = temp[2];

					if (pointarray[labelnum][xx].x < depthLeft)
						depthLeft = pointarray[labelnum][xx].x;
					if (pointarray[labelnum][xx].x > depthRight)
						depthRight = pointarray[labelnum][xx].x;
					if (pointarray[labelnum][xx].y < depthTop)
						depthTop = pointarray[labelnum][xx].y;
					if (pointarray[labelnum][xx].y > depthBottom)
						depthBottom = pointarray[labelnum][xx].y;
				}

				if ((depthRight - depthLeft)*(depthBottom - depthTop) >= 100) { // 2017.8.22 200에서 100으로 변경
					rectangle(depthLabelImage, Point(depthLeft, depthTop), Point(depthRight, depthBottom),
						Scalar(0, 0, 255), 2);

					//int x = centroids.at<double>(j, 0); //중심좌표
					//int y = centroids.at<double>(j, 1);
					//circle(color, Point(x, y), 5, Scalar(255, 0, 0), 1);
					string st = "Detect" + to_string(printtext++);;
					//putText(color, to_string(j), Point(left + 20, top + 20),FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
					putText(depthLabelImage, st, Point(depthLeft - 0, depthTop - 5), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255), 0.5);

					// CAMShift - depth 
					if (trackCounter2 == 1) {
						QcvCAMshiftTracker newTracker2;
						newTracker2.setCurrentRect(Rect(depthLeft, depthTop, depthRight - depthLeft, depthBottom - depthTop));
						camShiftTrackers2.push_back(newTracker2);
					}




				}



				//Rect crop(depthLeft - (depthLeft*0.3), depthTop - (depthLeft*0.3),
				//	(depthRight - depthLeft) + ((depthRight - depthLeft)*0.3),
				//	(depthBottom - depthTop) + ((depthBottom - depthTop)*0.3));

				//Rect crop(depthLeft -20 , depthTop-20,
				//	(depthRight - depthLeft) + 20,
				//	(depthBottom - depthTop) + 20);

				Rect crop(depthLeft , depthTop,
					(depthRight - depthLeft) ,
					(depthBottom - depthTop));

				if (crop.x <= 0)
					crop.x = depthLeft;
				if (crop.y <= 0)
					crop.y = depthTop;
				if (crop.width <= limage.cols)
					crop.width = depthRight - depthLeft;
				if (crop.height <= limage.rows)
					crop.height = depthBottom - depthTop;

				
				if ((depthRight - depthLeft)*(depthBottom - depthTop) >= 200) {
					Mat cropImage = limage(crop).clone(); // 별도메모리 사용 .clone()
					namedWindow("Cropping");
					moveWindow("Cropping", w * 0, h * 1);
					//resize(cropImage, cropImage, Size(), 2.0, 2.0);
					imshow("Cropping", cropImage);

				}
			
			
			}

		}

		QcvCAMshiftTracker::setMainImage(original);
		for (int trackText=1, i = 0; i < camShiftTrackers2.size(); i++)
		{
			if (camShiftTrackers2[i].trackCurrentRect().boundingRect().area() > 5)
			{
				//cv::ellipse(original, camShiftTrackers[i].trackCurrentRect(), cv::Scalar(0, 255, 0), 2, CV_AA);
				cv::rectangle(original, camShiftTrackers2[i].trackCurrentRect().boundingRect(), cv::Scalar(0, 0, 255), 2);
				circle(original, camShiftTrackers2[i].trackCurrentRect().center, 4, Scalar(0, 0, 255), -1);
				String tst = "OBJECT" + to_string(trackText++);
				putText(original, tst, 
					Point(camShiftTrackers2[i].trackCurrentRect().boundingRect().x,
						camShiftTrackers2[i].trackCurrentRect().boundingRect().y-5),
					FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0), 0.5);

				//line(original, camShiftTrackers[i-1].trackCurrentRect().center, camShiftTrackers[i].trackCurrentRect().center, Scalar(0, 0, 255), 1);
			}
		}
		if (trackCounter2 == 10) {
		
			camShiftTrackers2.clear();
			trackCounter2 = 0;
		}
		namedWindow("Color Segment");
		moveWindow("Color Segment", 1 * w, 3 * h);
		imshow("Color Segment", colorSegment);

		namedWindow("Depth Segment");
		moveWindow("Depth Segment", 1 * w, 4 * h);
		imshow("Depth Segment", color_depth);

		namedWindow("Depth Labeling Image");
		moveWindow("Depth Labeling Image", 0 * w, 4 * h);
		imshow("Depth Labeling Image", depthLabelImage);

		//namedWindow("Tracking");
		//moveWindow("Tracking", 0 * w, 1 * h);
		//imshow("Tracking", original);
	


		// fps bm 측정 끝
		time(&end);
		++counter;
		double sec = difftime(end, start);
		double fps = counter / sec;
		
		
		if (fpsi == 10) {
			cout << "FPS : " << fps << endl;
			fpsi = 0;
		}
		else {
			fpsi++;
		}

		//sprintf(strBuffer_bm, "%.2lf fps", fps_bm);
		//putText(disp8, strBuffer_bm, Point(10, 430), 1, 2, CV_RGB(0, 0, 255), 2);

	}
}
