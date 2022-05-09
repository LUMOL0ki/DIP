#pragma once
class Slic
{
public:
	Slic(cv::Mat src, int steps, int nc);

	static double getNumberOfSteps(int width, int height, double superpixels);

	void drawCenterGrid(cv::Mat& dst, cv::Scalar color);
	void drawContours(cv::Mat& dst, cv::Scalar color);
	void drawColorWithClusterMeans(cv::Mat& dst, cv::Scalar color);
private:

	std::vector<int> centersCount;

	std::vector<std::vector<int>> clusters;
	std::vector<std::vector<double>> distances;
	std::vector<std::vector<double>> centers;

	int step;
	int nc;
	int distance;

	void initializeData(cv::Mat src);
	void generateSuperpixels(cv::Mat& src, int steps, int nc);
	void createConectivity(cv::Mat& src);

	double calculateDistance(int clusterId, cv::Point pixel, cv::Scalar color);

	CvPoint findLocalMinimum(cv::Mat dst, cv::Point center);
};

