#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/aruco.hpp"
#include "Utils.h"
#include <vector>
using namespace cv;
using namespace std;

//cv::Mat cameraMatrix, distCoeffs;
std::vector<std::vector <double> > cameraMatrix;
std::vector<double> distCoeffs;


string tutorial_path = "/home/nimo/NewCamera/RealTimePoseEstimation_WithAurcoDetect_Data/";
string img_path = tutorial_path + "Regist/regist_orignal_1.jpg";
string video_path = tutorial_path + "box_with_60FPS.avi";

string project_path ="/home/nimo/NewCamera/ToolsOpenCV_Data/VideoToImg/";

// Some basic colors
Scalar red(0, 0, 255);
Scalar green(0,255,0);
Scalar blue(255,0,0);
Scalar yellow(0,255,255);

int main()
{
    try {
        VideoCapture capture;
        capture.open(video_path);
        if(!capture.isOpened())   // check if we succeeded
        {
            cout << "Could not open the camera device" << endl;
            return -1;
        }

        int frame_count = 0;
        Mat frame;
        while(capture.read(frame) && (char)waitKey(30) != 27) // capture frame until ESC is pressed
        {
            frame_count ++;
            cv::Mat inputImage = frame.clone();
            cv::Ptr<aruco::DetectorParameters> parameters;
            cv::Ptr<aruco::Dictionary> dictionary=aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f> > corners, rejectedCandidates;
            cv::aruco::detectMarkers(inputImage, dictionary, corners, ids);

            cout << ids.size() << "|" << corners.size() << endl;
            if (ids.size() > 0) {
                cv::aruco::drawDetectedMarkers(inputImage, corners, ids);
            }
            std::vector<cv::Point2f> imagePoints;
            int idsSize = ids.size();
            cout << "match points:" << idsSize << endl;
            drawText(inputImage, "Aruco Maker:" + IntToString(idsSize), green);
            imwrite(project_path + "frame_" + IntToString(frame_count) + ".jpg", inputImage);
//            imshow("show", inputImage);
//            waitKey(1);
        }
        destroyAllWindows();
    } catch (std::exception &ex) {
        cout<<"Exception :"<<ex.what()<<endl;
    }
}
