#include "stdafx.h"
#include "slic.h"

Slic::Slic(cv::Mat src, int numberOfSegments, int iterations)
{
	this->src = src;
	this->numberOfSegments = numberOfSegments;
	this->iterations = iterations;
	initializeCentroids(src, centroids);
}

void Slic::initializeCentroids(cv::Mat src, std::vector<cv::Point2d>& centroids)
{
	int step = std::sqrt((double)(src.cols * src.rows) / numberOfSegments);

	int offset = step / 2;

	int x = offset;
	int y = offset;

    for (int y = step / 2; y < src.rows; y += step) 
    {
        for (int x = step / 2; x < src.cols; x += step) 
        {
            cv::Vec3b color = src.at<cv::Vec3b>(y, x);
            cv::Point2d point(x, y);

            // Move the center to the lowest gradient position in a 3x3 neighborhood
            cv::Point2d min_point = point;
            double min_gradient = DBL_MAX;
            for (int dy = -1; dy <= 1; dy++) 
            {
                for (int dx = -1; dx <= 1; dx++) 
                {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= src.cols || ny >= src.rows) 
                    {
                        continue;
                    }
                    cv::Vec3b neighbor_color = src.at<cv::Vec3b>(ny, nx);
                    double dRGB = sqrt(pow(color[0] - neighbor_color[0], 2) +
                        pow(color[1] - neighbor_color[1], 2) +
                        pow(color[2] - neighbor_color[2], 2));
                    if (dRGB < min_gradient)
                    {
                        min_gradient = dRGB;
                        min_point = cv::Point2d(nx, ny);
                    }
                }
            }
            centroids.push_back(min_point);
        }
    }
}
