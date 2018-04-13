#ifndef QCVCAMSHIFTTRACKER_H
#define QCVCAMSHIFTTRACKER_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

class QcvCAMshiftTracker
{

public:
    QcvCAMshiftTracker();

    // Accessor functions ...
    static void setMainImage(const cv::Mat _mainImage);
    static cv::Mat getMainImage();

    void setCurrentRect(const cv::Rect _currentRect);
    cv::Rect getCurrentRect();

    cv::RotatedRect trackCurrentRect();

private:

    bool firstRun;

    static cv::Mat mainImage;

    cv::Rect currentRect;

    int VMin, VMax, SMin;

    cv::Rect trackWindow;
    int hsize;

    cv::Mat hsv, hue, mask, hist, backproj; //, histimg;

    cv::Point origin;
    cv::Rect selection;
    
};

#endif // QCVCAMSHIFTTRACKER_H
