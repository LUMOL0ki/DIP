#pragma once

struct ClusterCenter {
	cv::Point3i color;
	int R;
	int G;
	int B;
	cv::Point position;
	int x;
	int y;
};

class Slic
{
public:
	Slic(cv::Mat src, int numberOfSegments, int iterations);

	void computeSlic();
private:
	cv::Mat src;
	int S, numberOfSegments, iterations;
	std::vector<ClusterCenter> clusterCenters;

	void getBoundaries(int x, int y, int image_rows, int image_cols, int S, int& start_x, int& end_x, int& start_y, int& end_y);
	float euclideanDistance(ClusterCenter& a, ClusterCenter& b);
	cv::Point findLowestGradientPosition(cv::Mat& src, cv::Point clusterCenterPosition);
	cv::Point findLowestGradientPosition(cv::Mat& src, int x, int y);
	void getBoundaries(cv::Mat src, cv::Point clusterCenterPosition, cv::Point startPixel, cv::Point endPixel, int S);
	float recalculateClusters(cv::Mat& src, cv::Mat& indexer, std::vector<ClusterCenter>& clusterCenters);
};