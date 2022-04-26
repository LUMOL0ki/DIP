#include "centerOfTheMass.h"
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
	std::vector<cv::Point> objectIndexingColoredAndNumbered(cv::Mat src, cv::Mat& dst);

	void AssignText(int objectsCount, std::vector<cv::Point>& massCenters, cv::Mat& dst, std::vector<float>& centerOfMassAreas, std::vector<float>& circumferenceAreas);

	void etalonsClassification(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst);
	void kmeansClustering(cv::Mat src, cv::Mat& cl, cv::Mat& dst);
private:
	/// <summary>
	/// Indexing objects.
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Indexed destination image.</param>
	/// <returns>Number of objects.</returns>
	int objectIndexing(cv::Mat src, cv::Mat& dst);
	/// <summary>
	/// 
	/// </summary>
	/// <param name="src"></param>
	/// <param name="coordinateMoments"></param>
	/// <param name="perimeters"></param>
	/// <param name="count"></param>
	void getCoordinateMoments(cv::Mat src, std::vector<cv::Mat>& coordinateMoments, std::vector<float>& circumferenceMoments, int count);
	void getCenterOfMassMoments(cv::Mat& src, std::vector<cv::Mat>& coordinateMoments, std::vector<cv::Mat>& centerOfMassMoments);
	std::vector<cv::Point> getLookAroundMatrix(bool diagonal = true);
	void getClassificationMoments(std::vector<cv::Mat> centerOfMassMoments, std::vector<float> circumferenceMoments, std::vector<float>& F1, std::vector<float>& F2, std::vector<float>& uMin, std::vector<float>& uMax);
	const cv::Point2i& getPixelToCheck(cv::Point pixel, cv::Point neighbor);
	void initializeMoments(cv::Mat src, std::vector<cv::Mat>& coordinateMoments, std::vector<float>& circumferenceMoments, int count);
	void initializeCenterOfMassMoments(cv::Mat src, std::vector<cv::Mat> coordinateMoments, std::vector<cv::Mat>& centerOfMassMoments);
	void processIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, std::vector<cv::Point> lookAroundMatrix, int& id);
	void processCoordinate(std::vector<cv::Mat>& coordinateMoments, int id, int x, int y);
	void processCenterOfMass(std::vector<cv::Mat>& coordinateMoments, std::vector<cv::Mat>& centerOfMassMoments, int x, int y, int id);
	void assignIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, std::vector<cv::Point> lookAroundMatrix, int& id);
	/// <summary>
	/// Color me daddy.
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Destination image.</param>
	/// <param name="colors">array of colors</param>
	void assignRandomColors(cv::Mat src, cv::Mat& dst, int count);
	void calculateFeatures(float& F1s, float& F2s, float circumferenceMoment, float areaMoment, float uMins, float uMaxes);
	void calculateuMinuMax(float& uMin, float& uMax, cv::Mat centerOfTheMassMoment);
	float calculateArea(cv::Mat objectMoment);
	void calculateAreas(std::vector<cv::Mat> moments, std::vector<float>& momentsAreas);
	cv::Point calculateCenterOfMass(cv::Mat coordinateMoment);
	void calculateMassCenters(std::vector<cv::Mat>& coordinateMoments, std::vector<cv::Point>& massCenters);
	void featureExtraction(cv::Mat& indexedImg, std::vector<cv::Mat>& coordinateMoments, std::vector<float>& circumferenceAreas, int objectsCount, std::vector<cv::Mat>& centerOfMassMoments, std::vector<float>& F1, std::vector<float>& F2, std::vector<float>& uMin, std::vector<float>& uMax, std::vector<float>& coordinateAreas, std::vector<float>& centerOfMassAreas, std::vector<cv::Point>& massCenters);
	bool checkBoundaries(cv::Point pixel, cv::Mat src);
	std::vector<cv::Point> etalonsComputing(cv::Mat src, cv::Mat& dst, std::vector<float> F1, std::vector<float> F2, int count);
	void processEtalons(cv::Mat src, cv::Mat& dst, std::vector<cv::Point> massCenters, std::vector<cv::Point> etalons, std::vector<float>& F1, std::vector<float>& F2, int count);

	std::vector<cv::Point> kmeansComputing(cv::Mat src, cv::Mat& dst, std::vector<float> F1, std::vector<float> F2, int count);
	void processkmeans(cv::Mat scr, cv::Mat& dst, std::vector<cv::Point> massCenters, std::vector<cv::Point> kmeans, std::vector<float>& F1, std::vector<float>& F2, int count);
	/// <summary>
	/// Print features to console.
	/// </summary>
	/// <param name="centers"></param>
	/// <param name="coordinateAreas"></param>
	/// <param name="massAreas"></param>
	/// <param name="perimeters"></param>
	/// <param name="F1"></param>
	/// <param name="F2"></param>
	/// <param name="uMin"></param>
	/// <param name="uMax"></param>
	void printFeaturesToConsole(std::vector<cv::Point> massCenters, std::vector<float> coordinateAreas, std::vector<float> centerOfMassAreas, std::vector<float> circumferenceAreas, std::vector<float> F1s, std::vector<float> F2s, std::vector<float> uMins, std::vector<float> uMaxes);
	std::string generateText(int id, float value);
};

