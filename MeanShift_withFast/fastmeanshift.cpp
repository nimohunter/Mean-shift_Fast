#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "Utils.h"
#include <strstream>
#include <iostream>
#include <map>
using namespace cv;
using namespace std;

string project_path = "/home/nimo/NewCamera/MeanShift_withFast_Data/Result/";

string tutorial_path = "/home/nimo/NewCamera/RealTimePoseEstimation_WithAurcoDetect_Data/";
string video_path = tutorial_path + "box_with_60FPS.avi";

// Use for Fast Detect
int fast_threshold = 10;
vector<KeyPoint> fast_detect_key_points;

// Use for Aurco Detect
cv::Ptr<aruco::Dictionary> aruco_dictionary=aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
std::vector<int> ids, ids_pre;
std::vector<std::vector<cv::Point2f> > corners, corners_pre;
int aruco_nums, aruco_nums_pre;

// Use for float number equal
double min_threshold = 0.01;

// Remove absent
int max_num_absent_frame = 4;
map<int, int> aruco_marker_last_appear;
// Color
Scalar red(0, 0, 255);
Scalar green(0,255,0);
Scalar blue(255,0,0);
Scalar yellow(0,255,255);

void detectFastKeyPoint(Mat &img, vector<KeyPoint> &keyPoints);
double caculateAreaByShoelaceFormula(vector<Point2f> &vertices);
bool checkPointWithinParallelogram(Point2f point, vector<Point2f> &parallelogram_vertices, double parallelogram_area);
Point2f caculateTheCentroid(vector<Point2f> &keypoints_within);
void shiftAurcoMarker(int index, Point2f shift_vector);
Point2f calcShiftCenterToCentroid(int index);
void drawCenterAndEdge(Mat &_image, int index);
float getShiftVectorSize(Point2f point);
void calcAbsentMarkerIndex(vector <int> &collect);

void addEstimateDataFromPerFrame(int index, int now_frame);
void updateArucoMarkerShowMap(int frame_count);

int main()
{
    try {
        VideoCapture capture;
        capture.open(video_path);
        if(!capture.isOpened())
        {
            cout << "Could not open the camera device" << endl;
            return -1;
        }

        aruco_marker_last_appear.clear();
        aruco_nums_pre = 0;
        int frame_count = 0;
        Mat frame;
        while(capture.read(frame) && (char)waitKey(30) != 27) // capture frame until ESC is pressed
        {
            frame_count ++;
            Mat img_in_now = frame.clone();

            // step 1. get `img_in_now` fast detect
            Mat img_now = img_in_now.clone();
            detectFastKeyPoint(img_now, fast_detect_key_points);

            // Step 2. get Aruco detect
            cv::aruco::detectMarkers(img_now, aruco_dictionary, corners, ids);
            cv::aruco::drawDetectedMarkers(img_now, corners, ids);
            updateArucoMarkerShowMap(frame_count);
            aruco_nums = ids.size();

            vector <int> absent_marker_index_in_pre_frame;
            if (aruco_nums < aruco_nums_pre) {
                calcAbsentMarkerIndex(absent_marker_index_in_pre_frame);

                // Step3. detect the center and centroid
                for (int i = 0; i < absent_marker_index_in_pre_frame.size(); i++)
                {
                    Point2f shift_vector = Point2f(0.0, 0.0);
                    for (int iterat_id = 0; iterat_id < 10; iterat_id ++)
                    {
                        shiftAurcoMarker(absent_marker_index_in_pre_frame[i], shift_vector);
                        Point2f shift_vector_new = calcShiftCenterToCentroid(absent_marker_index_in_pre_frame[i]);

                        float new_shift_value = getShiftVectorSize(shift_vector_new);
                        float orignal_shift_value = getShiftVectorSize(shift_vector);

                        if (iterat_id > 0 && new_shift_value > orignal_shift_value ) {
//                            cout << "index:" << ids[i] << " count: " << iterat_id << endl;
                            break;
                        } else {
                            shift_vector = shift_vector_new;
                        }
                    }
                    drawCenterAndEdge(img_now, absent_marker_index_in_pre_frame[i]);
                    addEstimateDataFromPerFrame(absent_marker_index_in_pre_frame[i], frame_count);
                }

            }

            ids_pre = ids;
            corners_pre = corners;
            aruco_nums_pre = ids_pre.size();

            imwrite(project_path + "frame_" + IntToString(frame_count) + ".jpg", img_now);
        }
        destroyAllWindows();
    } catch (std::exception &ex) {
        cout<<"Exception :"<<ex.what()<<endl;
    }
}

//====================================================================
void updateArucoMarkerShowMap(int frame_count) {
    for (int i = 0; i < aruco_nums; i++) {
        aruco_marker_last_appear[ids[i]] = frame_count;
    }
}

void addEstimateDataFromPerFrame(int index, int now_frame) {
    if (now_frame < aruco_marker_last_appear[ids_pre[index]] + max_num_absent_frame) {
        ids.push_back(ids_pre[index]);
        corners.push_back(corners_pre[index]);
    }
}

void calcAbsentMarkerIndex(vector <int> &collect) {
    collect.clear();
    for (int i = 0; i < aruco_nums_pre; i++) {
        int id = ids_pre[i];
        bool isWithin = false;
        for (int j = 0; j < aruco_nums; j++) {
            if (ids[j] == id) {
                isWithin = true;
                break;
            }
        }
        if (!isWithin) {
            collect.push_back(i);
        }
    }
}

void drawCenterAndEdge(Mat &_image, int index) {

    vector<cv::Point2f> aruco_maker_vertices = corners_pre[index];
    for(int j = 0; j < 4; j++) {
        Point2f p0, p1;
        p0 = aruco_maker_vertices[j];
        p1 = aruco_maker_vertices[(j + 1) % 4];
        line(_image, p0, p1, blue, 1);
    }
    Point2f aruco_marker_center = caculateTheCentroid(aruco_maker_vertices);
    circle(_image, aruco_marker_center, 1, red);
}

float getShiftVectorSize(Point2f point) {
    return (point.x * point.x + point.y * point.y);
}

void shiftAurcoMarker(int index, Point2f shift_vector) {
    if (index >= ids_pre.size()) {
        return;
    }

    for (int i = 0; i < 4; i++) {
        corners_pre[index][i].x += shift_vector.x;
        corners_pre[index][i].y += shift_vector.y;
    }
}

Point2f caculateTheCentroid(vector<Point2f> &keypoints_within) {
    int this_size = keypoints_within.size();

    double x = 0.0, y = 0.0;
    for (int i = 0; i < this_size; i++) {
        x += keypoints_within[i].x;
        y += keypoints_within[i].y;
    }
    x /= this_size;
    y /= this_size;

    return Point2f(x, y);
}

Point2f calcShiftCenterToCentroid(int index) {
    if (index >= ids_pre.size()) {
        return Point2f(0.0, 0.0);
    }
//    int aruco_id = ids[index];
    vector<cv::Point2f> aruco_maker_vertices = corners_pre[index];

    vector<Point2f> keyPoint_within_aruco_marker;
    keyPoint_within_aruco_marker.clear();

    double parallelogram_area = caculateAreaByShoelaceFormula(aruco_maker_vertices);
    int fast_detect_key_points_size = fast_detect_key_points.size();

    for (int i = 0; i < fast_detect_key_points_size; i++) {
        Point2f one_point = fast_detect_key_points[i].pt;
        if (checkPointWithinParallelogram(one_point, aruco_maker_vertices, parallelogram_area)) {
            keyPoint_within_aruco_marker.push_back(one_point);
        }
    }

    Point2f aruco_marker_centroid = caculateTheCentroid(keyPoint_within_aruco_marker);
    Point2f aruco_marker_center = caculateTheCentroid(aruco_maker_vertices);
    return Point2f(aruco_marker_centroid.x - aruco_marker_center.x, aruco_marker_centroid.y - aruco_marker_center.y);
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
