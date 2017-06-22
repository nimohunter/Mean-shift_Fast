#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <strstream>
#include <iostream>
using namespace cv;
using namespace std;

string srcImage = "/home/nimo/NewCamera/MeanShift_withFast_Data/FastDetect/regist_orignal_5.jpg";

// Use for Fast Detect
int fast_threshold = 10;
vector<KeyPoint> fast_detect_key_points;

// Use for Aurco Detect
cv::Ptr<aruco::Dictionary> aruco_dictionary=aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
std::vector<int> ids;
std::vector<std::vector<cv::Point2f> > corners;

// Use for float number equal
double min_threshold = 0.01;

// Color
Scalar red(0, 0, 255);
Scalar green(0,255,0);
Scalar blue(255,0,0);
Scalar yellow(0,255,255);

void detectFastKeyPoint(Mat &img, vector<KeyPoint> &keyPoints);
double caculateAreaByShoelaceFormula(vector<Point2f> &vertices);
bool checkPointWithinParallelogram(Point2f point, vector<Point2f> &parallelogram_vertices, double parallelogram_area);
void handleOneArucoMakerWithMeanShift(int index, Mat _image);

int main()
{
    Mat img_in = imread(srcImage);

    // step 1. get fast detect
    Mat img_vis;
    img_in.copyTo(img_vis);
    detectFastKeyPoint(img_vis, fast_detect_key_points);

//    int size = fast_detect_key_points.size();
//    cout << "fast key:" << size << endl;
//    drawKeypoints(img_vis,fast_detect_key_points,img_vis,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2. get Aruco detect
    img_in.copyTo(img_vis);
    cv::aruco::detectMarkers(img_vis, aruco_dictionary, corners, ids);


//    if (ids.size() > 0) {
//    cv::aruco::drawDetectedMarkers(img_vis, corners, ids);
//    }
//    imshow("dsd", img_vis);
//    waitKey(0);
    handleOneArucoMakerWithMeanShift(0, img_vis);



    return 0;
}

//====================================================================
void handleOneArucoMakerWithMeanShift(int index, Mat _image) {
    if (index >= ids.size()) {
        return;
    }
    int aruco_id = ids[index];
    vector<cv::Point2f> aruco_maker_vertices = corners[index];

    for(int j = 0; j < 4; j++) {
        Point2f p0, p1;
        p0 = aruco_maker_vertices[j];
        p1 = aruco_maker_vertices[(j + 1) % 4];
        line(_image, p0, p1, green, 1);
    }

    double parallelogram_area = caculateAreaByShoelaceFormula(aruco_maker_vertices);
    int fast_detect_key_points_size = fast_detect_key_points.size();
    for (int i = 0; i < fast_detect_key_points_size; i++) {
        Point2f one_point = fast_detect_key_points[i].pt;
        if (checkPointWithinParallelogram(one_point, aruco_maker_vertices, parallelogram_area)) {
            circle(_image, one_point, 1, red);
        } else {
            circle(_image, one_point, 1, yellow);
        }
    }


    imshow("show", _image);
    waitKey(0);
}

void detectFastKeyPoint(Mat &img, vector<KeyPoint> &keyPoints) {
    Ptr<FastFeatureDetector> fastDetector = FastFeatureDetector::create(fast_threshold, true, FastFeatureDetector::TYPE_9_16);
    fastDetector ->detect(img, keyPoints);
}

bool checkPointWithinParallelogram(Point2f point, vector<Point2f> &parallelogram_vertices, double parallelogram_area) {
    int parallelogram_vertices_size = parallelogram_vertices.size();
    if (parallelogram_vertices_size != 4) {
        // Only support parallelogram
        return false;
    }

    vector<Point2f> triangle_vertices;
    double total_area = 0.0;
    bool result_whthin = true;
    for (int i = 0; i < parallelogram_vertices_size; i++) {
        triangle_vertices.clear();
        triangle_vertices.push_back(point);
        triangle_vertices.push_back(parallelogram_vertices[i]);
        triangle_vertices.push_back(parallelogram_vertices[(i + 1) % parallelogram_vertices_size]);
        total_area += caculateAreaByShoelaceFormula(triangle_vertices);

        if (total_area > parallelogram_area + min_threshold) {
            result_whthin = false;
            break;
        }
    }

    return result_whthin;
}

double caculateAreaByShoelaceFormula(vector<Point2f> &vertices) {
    int vertices_size = vertices.size();
    if (vertices_size < 3) {
        return 0;
    }

    double area_result = 0.0;
    for (int i = 0; i < vertices_size; i++) {
        area_result += vertices[i].x * vertices[(i + 1) % vertices_size].y;
        area_result -= vertices[i].y * vertices[(i + 1) % vertices_size].x;
    }

    if (area_result < 0) {
        area_result = 0;
    }

    return area_result;
}