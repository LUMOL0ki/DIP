#pragma once

struct Boundary
{
	float uMin;
	float uMax;
};

class Object
{
public:
	/// <summary>
	/// Initializes a new object.
	/// </summary>
	/// <param name="id"></param>
	/// <param name="src">source image.</param>
	Object(int id, cv::Mat src);
	/// <summary>
	/// Get id of this object.
	/// </summary>
	/// <returns></returns>
	int getId();

	cv::Mat getCoordinateMoment();
	cv::Mat getCenterOfMassMoment();
	cv::Point2f getCenterOfMass();

	float getCoordinateArea();
	float getCenterOfMassArea();
	float getCircumferenceArea();
	float getFirstFeature();
	float getSecondFeature();

	Boundary getBoundary();
	/// <summary>
	/// Get number of object in image.
	/// </summary>
	/// <param name="src"></param>
	/// <param name="dst"></param>
	/// <returns></returns>
	static int getNumberOfObjects(cv::Mat src, cv::Mat& dst);
private:
	int id;

	cv::Mat coordinateMoment;
	cv::Mat centerOfMassMoment;
	cv::Point2f centerOfMass;

	float coordinateArea;
	float centerOfMassArea;
	float circumferenceArea;
	float F1;
	float F2;

	Boundary boundary;

	float getAreaFromMoment(cv::Mat objectMoment);
	void initializeCoordinate(cv::Mat src);
	void initializeCenterOfMass(cv::Mat src);
	void initializeCircumference();
	void processCoordinate(int x, int y);
	void processCenterOfMass(int x, int y);
	void calculateCenterOfMass();
	void calculateuMinuMax();
	void calculateFeatures();

	static bool checkBoundaries(cv::Point pixel, cv::Mat src);
	static const cv::Point getPixelToCheck(cv::Point pixel, cv::Point neighbor);
	static void initializeObjectsCounting(cv::Mat& src, cv::Mat& dst, int& id);
	static void lookForObject(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const std::vector<cv::Point> lookAroundMatrix, int& id);
};

