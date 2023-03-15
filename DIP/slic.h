#pragma once
class Slic
{
public:
	Slic(cv::Mat src, int numberOfSegments, int iterations);

private:
	cv::Mat src;
	int S, numberOfSegments, iterations;
	std::vector<cv::Point2d> centroids;

	void initializeCentroids(cv::Mat src, std::vector<cv::Point2d>& centroids);
};