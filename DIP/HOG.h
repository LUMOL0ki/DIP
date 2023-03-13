#pragma once
class HOG
{
public:
	HOG(cv::Mat src);

	void compute(cv::Mat& gradientImage, cv::Mat& orientationImage);
	void createHistograms(cv::Mat gradientImage, cv::Mat orientationImage, int blockSize, int cellSize, int numBins, cv::Mat& histograms);

private:
	cv::Mat src;
	cv::Mat graySrc;
	//cv::Mat gx, gy;

	void computeOrientationOfGradient(cv::Mat src, cv::Mat gx, cv::Mat gy, cv::Mat& orientationImage);
	void computeSizeOfGradient(cv::Mat src, cv::Mat gx, cv::Mat gy, cv::Mat& gradientImage);
	void brightnessDifference(cv::Mat src, cv::Mat& gx, cv::Mat& gy);
};

