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

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	// left video size > right video size
	VideoCapture leftCap("E:/1. cctv data (video)/20170904/Left (1).avi"); // open left video 
	VideoCapture rightCap("E:/1. cctv data (video)/20170904/Right (1).avi"); // open right video 
	

	if (!leftCap.isOpened())
	{
		cout << "Cannot open the left video file" << endl;
		return -1;
	}
	if (!rightCap.isOpened())
	{
		cout << "Cannot open the right video file" << endl;
		return -1;
	}

	double leftFps = leftCap.get(CV_CAP_PROP_FPS);
	cout << "first : Frame per seconds : " << leftFps << endl;

	double rightFps = leftCap.get(CV_CAP_PROP_FPS);
	cout << "first : Frame per seconds : " << rightFps << endl;

	namedWindow("left video", CV_WINDOW_AUTOSIZE);
	namedWindow("right video", CV_WINDOW_AUTOSIZE);




	Mat lgray, rgray;
	Mat beforeLImage, beforeRImage;
	Rect rect(0, 250, 800 - 0, 450 - 250);

	// MOG2 : Background Substraction
	Mat mog;
	Ptr <BackgroundSubtractorMOG2> pMOG2;
	pMOG2 = createBackgroundSubtractorMOG2(600, 16, true);

	// Mophology
	Mat element(5, 5, CV_8U, Scalar(1));

	// connected ComponentsWithstats
	Mat binary, color;
	Mat labels, stats, centroids;

	// StereoSGBM parameter
	int ndisparities = 16 * 7;   // Range of disparity, 윈도우 사이즈 클수록 크게 지정
	int SADWindowSize = 11; // Size of the block window. Must be odd, 윈도우 사이즈 클수록 크게 지정
	
	// StereoSGBM create
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, ndisparities, SADWindowSize);
	int P1 = 200, P2 = 4000;
	sgbm->setP1(P1);
	sgbm->setP2(P2);

	char savefile[300] ="C:/Users/박서희/Documents/Visual Studio 2015/Projects/ConsoleApplication1/ConsoleApplication1/imagecapture/";
	int savenum = 0;

	while (1)
	{
		//Mat beforeLimage, beforeRimage;
		Mat limage, rimage;
		//int w = 800 * 0.8, h = (450 - 250)*0.8 + 100;
		int w = 800, h = (450 - 250) + 100;
		
		bool bSuccess1 = leftCap.read(limage);
		bool bSuccess2 = rightCap.read(rimage);

		if (!bSuccess1) 
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
		if (!bSuccess2) 
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}


		//limage = beforeLImage(rect);
		//resize(limage, limage, Size(), 0.8, 0.8);
		namedWindow("left video");
		moveWindow("left video", 0 * w, 0 * h);
		imshow("left video", limage);

		//rimage = beforeRImage(rect);
		//resize(rimage, rimage, Size(), 0.8, 0.8);
		namedWindow("right video");
		moveWindow("right video", 1 * w, 0 * h);
		imshow("right video", rimage);

		if (waitKey(leftFps) == 0){
			cout << "esc key is pressed by user" << endl;
			break;
		}
		if (waitKey(rightFps) == 0) {
			cout << "esc key is pressed by user" << endl;
			break;
		}

		//cout << "Frame per seconds : " << leftFps << endl;
		//cout << "Frame per seconds : " << rightFps << endl;


		cvtColor(limage, lgray, CV_BGR2GRAY);
		cvtColor(rimage, rgray, CV_BGR2GRAY);

		// mog 매개변수 true 일 때 
		GaussianBlur(lgray, lgray, Size(7, 7), 1.5, 1.5);
		pMOG2->apply(lgray, mog);
		//namedWindow("Original MOG");
		//moveWindow("Original MOG", 0 * w, 2 * h);
		//imshow("Original MOG", mog);

		threshold(mog, mog, 30, 255, THRESH_BINARY);
		//imshow("ddddd", mog);

		dilate(mog, mog, Mat(), Point(-1, -1), 1);
		erode(mog, mog, Mat(), Point(-1, -1), 1);
		dilate(mog, mog, Mat(), Point(-1, -1), 1);
		//threshold(mog, mog, 200, 255, THRESH_BINARY);

		//namedWindow("MOG");
		//moveWindow("MOG", 1 * w, 2 * h);
		//imshow("MOG", mog);

		// object labeling : MOG로 객체 추출 한 것을 라벨링 하는 과정
		threshold(mog, binary, 127, 255, THRESH_BINARY);

		cvtColor(binary, color, COLOR_GRAY2BGR);

		int numOfLables = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
		int n = 0; // Group number

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


			}

		}

		namedWindow("object labeling");
		moveWindow("object labeling", 0 * w , 2 * h -200);
		imshow("object labeling", color);






		Mat left_disp, right_disp;
		sgbm->compute(limage, rimage, left_disp);

		Mat imgDisparity8U = Mat(limage.rows, limage.cols, CV_8UC1);
		cv::normalize(left_disp, imgDisparity8U, 0, 255, CV_MINMAX, CV_8U);

		double filtering_time = (double)getTickCount();
		Mat filtered_disp;

		filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();

		Mat raw_disp_vis;
		cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, 2);


		cv::namedWindow("raw disparity");
		cv::moveWindow("raw disparity", 1 * w, 1 * h);
		cv::imshow("raw disparity", raw_disp_vis);







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
						segmentPos.push_back(Rect(x, y, 0, 0));
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
			int depthLeft = INT_MAX, depthRight = 0, depthWidth = 0, depthHeight = 0, depthTop = INT_MAX, depthBottom = 0;

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


					Rect crop(depthLeft, depthTop, (depthRight - depthLeft), (depthBottom - depthTop));

					if (crop.x <= 0)
						crop.x = depthLeft;
					if (crop.y <= 0)
						crop.y = depthTop;
					if (crop.width <= limage.cols)
						crop.width = depthRight - depthLeft;
					if (crop.height <= limage.rows)
						crop.height = depthBottom - depthTop;


					if ((depthRight - depthLeft)*(depthBottom - depthTop) >= 500) {
						Mat cropImage = limage(crop).clone(); // 별도메모리 사용 .clone()


						int cropWidth = cropImage.cols;
						int cropHeight = cropImage.rows;
						int target_width = 100;

						Mat cropSquare = Mat::zeros(target_width, target_width, cropImage.type());

						int max_dim = (cropWidth >= cropHeight) ? cropWidth : cropHeight;
						float cropScale = ((float)target_width) / max_dim;

						Rect cropROI;

						if (cropWidth >= cropHeight) {
							cropROI.width = target_width;
							cropROI.x = 0;
							cropROI.height = cropHeight * cropScale;
							cropROI.y = (target_width - cropROI.height) / 2;
						}
						else {
							cropROI.y = 0;
							cropROI.height = target_width;
							cropROI.width = cropWidth * cropScale;
							cropROI.x = (target_width - cropROI.width) / 2;
						}

						resize(cropImage, cropSquare(cropROI), cropROI.size());




						namedWindow("Cropping");
						moveWindow("Cropping", w * 0, h * 1);
						//resize(cropImage, cropImage, Size(), 2.0, 2.0, INTER_CUBIC);
						imshow("Cropping", cropSquare);


						sprintf(savefile, "original%d.jpg", savenum++);
						imwrite(savefile, limage);
						//waitKey(100);



					}
				}

/// 여기자리 


			}

		}

		namedWindow("Color Segment");
		moveWindow("Color Segment", 1 * w, 2 * h);
		imshow("Color Segment", colorSegment);

		namedWindow("Depth Segment");
		moveWindow("Depth Segment", 1 * w, 3 * h);
		imshow("Depth Segment", color_depth);

		namedWindow("Depth Labeling Image");
		moveWindow("Depth Labeling Image", 0 * w, 3 * h-200);
		imshow("Depth Labeling Image", depthLabelImage);



	}
	return 0;
}