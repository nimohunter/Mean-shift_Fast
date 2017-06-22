#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <strstream>
using namespace cv;
using namespace std;

string srcImage = "/home/nimo/NewCamera/ToolsOpenCV_Data/FastDetect/regist_orignal_5.jpg";

int main()
{
    Mat img = imread(srcImage);

    std::vector<KeyPoint> keyPoints;

    Ptr<FastFeatureDetector> fastDetector = FastFeatureDetector::create(10, true, FastFeatureDetector::TYPE_9_16);
    fastDetector ->detect(img, keyPoints);

    Mat img_out;
    drawKeypoints(img,keyPoints,img_out,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keyPoint image",img_out);
    waitKey(0);

    return 0;
}
