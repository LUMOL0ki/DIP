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
	
	std::vector<Object> etalonsClassification(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst);
	void kmeansClustering(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst);
	std::vector<Object> ExteractObjects(cv::Mat src);
private:
	/// <summary>
	/// Color me daddy.
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Destination image.</param>
	/// <param name="colors">array of colors</param>
	void assignRandomColors(cv::Mat src, cv::Mat& dst, int count);
	int featureExtraction(cv::Mat src, cv::Mat& dst, std::vector<Object>& objects);
	
	std::vector<cv::Point2f> etalonsComputing(cv::Mat src, cv::Mat& dst, std::vector<Object>& objects);
	
	void drawEtalons(std::vector<cv::Point2f>& etalons, cv::Mat& dst);
	void processEtalons(cv::Mat src, cv::Mat& dst, std::vector<Object>& objects, std::vector<cv::Point2f> etalons);

	void assignIds(std::vector<cv::Point2f> srcClasses, cv::Point2f pixel, float& currentDistance, int& currentId);

	double calculateDistance(float x, float y);
	void calculateEtalon(cv::Point2f& etalon, int Nr, float etalonF1, float etalonF2);

	std::vector<cv::Point2f> kmeansComputing(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, int& steps, int numberOfClusters = 3);
	void recalculateCentroid(int numOfPixels, cv::Point2f& centroid, float meanX, float meanY);
	void assignIds(std::vector<int>& kmeansIds, std::vector<int>& newkmeansIds, bool& isNewCentroid);
	void checkDistance(float srcDistance, float& dstDistance, float newId, int& currentId);
	void initializePixels(std::vector<Object>& objects, std::vector<cv::Point2f>& pixels, cv::Mat& dst);
	void processkmeans(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, std::vector<cv::Point2f> kmeans);


	void initializeCentroids(int numberOfClusters, std::vector<cv::Point2f>& centroids);
	void drawCentroids(std::vector<cv::Point2f> centroids, cv::Mat& dst);

	void drawCenterOfObject(Object& object, cv::Mat& dst, cv::Vec3f color);
	/// <summary>
	/// /// Print features to console.
	/// </summary>
	/// <param name="objects"></param>
	void printFeaturesToConsole(std::vector<Object> objects);
	std::string generateText(int id, float value);
	void AssignText(std::vector<Object> objects, cv::Mat& dst);
	void convertColorToGray32(cv::Mat src, cv::Mat& dst);
};

