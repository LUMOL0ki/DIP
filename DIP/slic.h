#pragma once

struct ClusterCenter {
	cv::Point3i color;
	cv::Point position;
};

class Slic
{
public:
	Slic(cv::Mat src, int numberOfSegments, int iterations = 5, float threshold = 2.0f);

	void computeSlic();
private:
	cv::Mat src;
	int S, numberOfSegments, iterations, threshold;
	std::vector<ClusterCenter> clusterCenters;

	void initializeClusterCenters(cv::Mat src, cv::Mat& regularIntervals);
	float initializeSteps(int pixelSize, int numberOfSegments);
	void moveClusterCentersToLowestGradient(cv::Mat src, cv::Mat& lowestGradients);
	float euclideanDistance(ClusterCenter& a, ClusterCenter& b);
	cv::Point findLowestGradientPosition(cv::Mat& src, ClusterCenter clusterCenter);
	void getBoundaries(cv::Mat src, ClusterCenter clusterCenter, cv::Point& start, cv::Point& end, int S);
	float recalculateClusters(cv::Mat& src, cv::Mat& indexer, std::vector<ClusterCenter>& clusterCenters);
};