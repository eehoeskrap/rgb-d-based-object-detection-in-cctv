#include "qcvcamshifttracker.h"

// Static member definition ...
cv::Mat QcvCAMshiftTracker::mainImage;

QcvCAMshiftTracker::QcvCAMshiftTracker()
{

    hsize = 16;

    VMin = 10;
    VMax = 256;
    SMin = 30;

    firstRun = true;

//    histimg = cv::Mat::zeros(200, 320, CV_8UC3);

}

void QcvCAMshiftTracker::setMainImage(const cv::Mat _mainImage)
{
    _mainImage.copyTo(mainImage);
}

cv::Mat QcvCAMshiftTracker::getMainImage()
{
    return mainImage;
}

void QcvCAMshiftTracker::setCurrentRect(const cv::Rect _currentRect)
{
    currentRect = _currentRect;
    selection = currentRect;
}

cv::Rect QcvCAMshiftTracker::getCurrentRect()
{
    return currentRect;
}

cv::RotatedRect QcvCAMshiftTracker::trackCurrentRect()
{

    cv::RotatedRect trackBox;

    float hranges[] = {0,180};
    const float* phranges = hranges;

    cv::Mat image;
    mainImage.copyTo(image);

    cv::cvtColor(image, hsv, CV_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(0, SMin, MIN(VMin,VMax)),
                cv::Scalar(100, 256, MAX(VMin, VMax)), mask);
	
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

	//imshow("HSV", hsv);

    if( firstRun )
    {
        cv::Mat roi(hue, selection), maskroi(mask, selection);
        cv::calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        cv::normalize(hist, hist, 0, 255, CV_MINMAX);

        trackWindow = selection;
        firstRun = false;

//        histimg = cv::Scalar::all(0);
//        int binW = histimg.cols / hsize;
//        cv::Mat buf(1, hsize, CV_8UC3);
//        for( int i = 0; i < hsize; i++ )
//            buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180./hsize), 255, 255);
//        cv::cvtColor(buf, buf, CV_HSV2BGR);

//        for( int i = 0; i < hsize; i++ )
//        {
//            int val = cv::saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
//            cv::rectangle( histimg, cv::Point(i*binW,histimg.rows),
//                           cv::Point((i+1)*binW,histimg.rows - val),
//                           cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8 );
//        }

    }

    cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);

	//imshow("backproj", backproj);

    backproj &= mask;

    trackBox = cv::CamShift(backproj, trackWindow,
                            cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

    //        if( trackWindow.area() <= 1 )
    //        {
    //            int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
    //            trackWindow = cv::Rect(trackWindow.x - r, trackWindow.y - r,
    //                               trackWindow.x + r, trackWindow.y + r) &
    //                          cv::Rect(0, 0, cols, rows);
    //        }

    //        if( backprojMode )
    //            cv::cvtColor( backproj, image, CV_GRAY2BGR );

    return trackBox;

}
