#include "centerOfTheMass.h"
#include "object.h"
#pragma once
class FeaturesDetector
{
public:
	/// <summary>
	/// Thresholding the Image.
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Destination image.</param>
	/// <param name="threshold">Threshold value.</param>
	void imageThresholding(cv::Mat src, cv::Mat& dst, float threshold);
	/// <summary>
	/// Indexing and coloring destination image.
	/// </summary>
	/// <param name="src"></param>
	/// <param name="dst"></param>
	void objectIndexingColored(cv::Mat src, cv::Mat& dst);
	/// <summary>
	/// Indexing, coloring and numbering destination image. 
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Indexed destination image.</param>
	/// <returns></returns>
	void objectIndexingColoredAndNumbered(cv::Mat src, cv::Mat& dst);

	void AssignText(std::vector<Object> objects, cv::Mat& dst);
	
	void etalonsClassification(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst);
	void kmeansClustering(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst);
private:
	/// <summary>
	/// Color me daddy.
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Destination image.</param>
	/// <param name="colors">array of colors</param>
	void assignRandomColors(cv::Mat src, cv::Mat& dst, int count);
	int featureExtraction(cv::Mat src, cv::Mat& dst, std::vector<Object>& objects);
	
	std::vector<cv::Point2f> etalonsComputing(cv::Mat src, cv::Mat& dst, std::vector<Object> objects);
	void processEtalons(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, std::vector<cv::Point2f> etalons);
	
	std::vector<cv::Point2f> kmeansComputing(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, int numberOfClusters = 3, int steps = 0);
	void processkmeans(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, std::vector<cv::Point2f> kmeans);
	/// <summary>
	/// /// Print features to console.
	/// </summary>
	/// <param name="objects"></param>
	void printFeaturesToConsole(std::vector<Object> objects);
	std::string generateText(int id, float value);
};

