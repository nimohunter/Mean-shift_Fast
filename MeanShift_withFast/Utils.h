/*
 * Utils.h
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include "opencv2/opencv.hpp"

// Draw a text with the question point
void drawQuestion(cv::Mat image, cv::Point3f point, cv::Scalar color);

// Draw a text with the number of entered points
void drawText(cv::Mat image, std::string text, cv::Scalar color);

// Draw a text with the number of entered points
void drawText2(cv::Mat image, std::string text, cv::Scalar color);

// Draw a text with the frame ratio
void drawFPS(cv::Mat image, double fps, cv::Scalar color);

// Draw a text with the frame ratio
void drawConfidence(cv::Mat image, double confidence, cv::Scalar color);

// Draw a text with the number of entered points
void drawCounter(cv::Mat image, int n, int n_max, cv::Scalar color);

// Draw the points and the coordinates
void drawPoints(cv::Mat image, std::vector<cv::Point2f> &list_points_2d, std::vector<cv::Point3f> &list_points_3d, cv::Scalar color);

// Draw only the 2D points
void draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color);

// Draw an arrow into the image
void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0);

// Draw the 3D coordinate axes
void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d);


// Converts a given string to an integer
int StringToInt ( const std::string &Text );

// Converts a given float to a string
std::string FloatToString ( float Number );

// Converts a given integer to a string
std::string IntToString ( int Number );


#endif /* UTILS_H_ */
