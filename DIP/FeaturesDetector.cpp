#include "stdafx.h"
#include "FeaturesDetector.h"
#include "colorHelper.h"

void FeaturesDetector::imageThresholding(cv::Mat src, cv::Mat& dst, float threshold)
{
	dst = cv::Mat::zeros(src.size(), src.type()); // init destination with zeros (black).
	for (int y = 0; y < src.rows; y++) 
	{
		for (int x = 0; x < src.cols; x++) 
		{
			if (src.at<float>(y, x) > threshold) 
			{
				dst.at<float>(y, x) = 1; // assign one (white).
			}
		}
	}
}

void FeaturesDetector::objectIndexingColored(cv::Mat src, cv::Mat& dst)
{
	cv::Mat indexedImg = cv::Mat::zeros(src.size(), src.type());
	dst = cv::Mat::zeros(src.size(), CV_32FC3);

	int objectsCount = objectIndexing(src, indexedImg);
	assignRandomColors(indexedImg, dst, objectsCount);
}

std::vector<cv::Point> FeaturesDetector::objectIndexingColoredAndNumbered(cv::Mat src, cv::Mat& dst)
{
	std::vector<float> coordinateAreas;
	std::vector<float> centerOfMassAreas;
	std::vector<float> circumferenceAreas;
	std::vector<float> F1;
	std::vector<float> F2;
	std::vector<float> uMin;
	std::vector<float> uMax;
	cv::Mat indexedImg = cv::Mat::zeros(src.size(), src.type());
	dst = cv::Mat::zeros(src.size(), CV_32FC3);
	std::vector<cv::Mat> coordinateMoments;
	std::vector<cv::Mat> centerOfMassMoments;
	std::vector <cv::Point> massCenters;

	int objectsCount = objectIndexing(src, indexedImg);
	assignRandomColors(indexedImg, dst, objectsCount);

	featureExtraction(
		indexedImg,
		coordinateMoments,
		circumferenceAreas,
		objectsCount,
		centerOfMassMoments,
		F1, F2,
		uMin, uMax,
		coordinateAreas,
		centerOfMassAreas,
		massCenters);
	
	AssignText(objectsCount, massCenters, dst, centerOfMassAreas, circumferenceAreas);	
	printFeaturesToConsole(massCenters, coordinateAreas, centerOfMassAreas, circumferenceAreas, F1, F2, uMin, uMax);

	return massCenters;
	
}

void FeaturesDetector::AssignText(int objectsCount, std::vector<cv::Point>& massCenters, cv::Mat& dst, std::vector<float>& centerOfMassAreas, std::vector<float>& circumferenceAreas)
{

	cv::Scalar white = cv::Scalar(1.0, 1.0, 1.0);
	for (int i = 0; i < objectsCount; i++) {
		int id = i + 1;
		cv::Point centerOfMass = massCenters[i];
		cv::putText(dst, generateText(id, centerOfMassAreas[i]), cv::Point(centerOfMass.y - 12, centerOfMass.x), cv::FONT_HERSHEY_SIMPLEX, 0.3, white);
		cv::putText(dst, generateText(id, circumferenceAreas[i]), cv::Point(centerOfMass.y - 12, centerOfMass.x + 11), cv::FONT_HERSHEY_SIMPLEX, 0.3, white);
	}
}

void FeaturesDetector::etalonsClassification(cv::Mat src, cv::Mat& cl, cv::Mat& dst)
{
	std::vector<float> coordinateAreas;
	std::vector<float> centerOfMassAreas;
	std::vector<float> circumferenceAreas;
	std::vector<float> F1;
	std::vector<float> F2;
	std::vector<float> uMin;
	std::vector<float> uMax;
	cv::Mat indexedImg = cv::Mat::zeros(src.size(), src.type());
	dst = cv::Mat::zeros(src.size(), CV_32FC3);
	std::vector<cv::Mat> coordinateMoments;
	std::vector<cv::Mat> centerOfMassMoments;
	std::vector <cv::Point> massCenters;

	int objectsCount = objectIndexing(src, indexedImg);
	featureExtraction(
		indexedImg, 
		coordinateMoments, 
		circumferenceAreas, 
		objectsCount, 
		centerOfMassMoments, 
		F1, F2, 
		uMin, uMax, 
		coordinateAreas, 
		centerOfMassAreas, 
		massCenters);

	etalonsComputing(src, dst, F1, F2, objectsCount);
}

std::vector<cv::Point> FeaturesDetector::getLookAroundMatrix(bool diagonal)
{
	if (diagonal) 
	{
		return {
			cv::Point(1, 1), cv::Point(1, 0), cv::Point(1, -1),
			cv::Point(0, 1), /* you are here */	cv::Point(0, -1),
			cv::Point(-1, 1), cv::Point(-1, 0), cv::Point(-1, 1)
		};
	}
	else 
	{
		return {
							 cv::Point(1, 0),
			cv::Point(0, 1), /* you are here */	cv::Point(0, -1),
							 cv::Point(-1, 0)
		};
	}
}

int FeaturesDetector::objectIndexing(cv::Mat src, cv::Mat& dst)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	std::vector<cv::Point> pixels;
	std::vector<cv::Point> lookAroundMatrix = getLookAroundMatrix();
	int id = 1;

	processIndexes(src, dst, pixels, lookAroundMatrix, id);

	return id - 1; // minus backgroung;
}

void FeaturesDetector::getCoordinateMoments(cv::Mat src, std::vector<cv::Mat>& coordinateMoments, std::vector<float>& circumferenceMoments, int count)
{
	initializeMoments(src, coordinateMoments, circumferenceMoments, count);
	std::vector<cv::Point> lookAroundMatrix = getLookAroundMatrix(false);
	int lookAroundMatrixSize = (lookAroundMatrix.size() * sizeof(cv::Point)) / sizeof(cv::Point);

	for (int y = 0; y < src.rows; y++) 
	{
		for (int x = 0; x < src.cols; x++) 
		{
			int id = src.at<float>(y, x);

			if (id != 0.0f) 
			{
				processCoordinate(coordinateMoments, id, x, y);

				cv::Point pixel = cv::Point(x, y);

				if (checkBoundaries(pixel, src))
				{
					int circumferenceCounter = 0;

					for (cv::Point neighbor : lookAroundMatrix) 
					{
						cv::Point pixelToCheck = getPixelToCheck(pixel, neighbor);
						if (src.at<float>(pixelToCheck) == id)
						{
							circumferenceCounter++;
						}
					}

					if (circumferenceCounter != lookAroundMatrixSize) 
					{
						circumferenceMoments[id-1]++;
					}
				}
			}
		}
	}
}

void FeaturesDetector::getCenterOfMassMoments(cv::Mat& src, std::vector<cv::Mat>& coordinateMoments, std::vector<cv::Mat>& centerOfMassMoments)
{
	initializeCenterOfMassMoments(src, coordinateMoments, centerOfMassMoments);

	for (int y = 0; y < src.rows; y++) 
	{
		for (int x = 0; x < src.cols; x++) 
		{
			int id = src.at<float>(y, x);

			if (id != 0.0f) 
			{
				processCenterOfMass(coordinateMoments, centerOfMassMoments, x, y, id - 1);
			}
		}
	}
}

void FeaturesDetector::getClassificationMoments(std::vector<cv::Mat> centerOfMassMoments, std::vector<float> circumferenceMoments, std::vector<float>& F1s, std::vector<float>& F2s, std::vector<float>& uMins, std::vector<float>& uMaxes)
{
	if (centerOfMassMoments.size() == circumferenceMoments.size())
	{
		for (int i = 0; i < centerOfMassMoments.size(); i++)
		{
			float F1, F2;
			float uMin, uMax;

			calculateuMinuMax(uMin, uMax, centerOfMassMoments[i]);
			calculateFeatures(F1, F2, circumferenceMoments[i], calculateArea(centerOfMassMoments[i]), uMin, uMax);
			F1s.push_back(F1);
			F2s.push_back(F2);
			uMins.push_back(uMin);
			uMaxes.push_back(uMax);
		}
	}
}

const cv::Point2i& FeaturesDetector::getPixelToCheck(cv::Point pixel, cv::Point neighbor)
{
	return pixel + neighbor;
}

void FeaturesDetector::calculateAreas(std::vector<cv::Mat> moments, std::vector<float>& momentsAreas)
{
	for (cv::Mat moment : moments) {
		momentsAreas.push_back(calculateArea(moment));
	}
}

void FeaturesDetector::initializeMoments(cv::Mat src, std::vector<cv::Mat>& coordinates, std::vector<float>& circumferences, int count)
{
	for (int i = 0; i < count; i++) {
		coordinates.push_back(cv::Mat::zeros(cv::Size(2, 2), src.type()));
		circumferences.push_back(0.0f);
	}
}

void FeaturesDetector::initializeCenterOfMassMoments(cv::Mat src, std::vector<cv::Mat> coordinateMoments, std::vector<cv::Mat>& centerOfMassMoments)
{
	for (int i = 0; i < coordinateMoments.size(); i++) {
		centerOfMassMoments.push_back(cv::Mat::zeros(cv::Size(3, 3), src.type()));
	}
}

void FeaturesDetector::processCenterOfMass(std::vector<cv::Mat>& coordinateMoments, std::vector<cv::Mat>& centerOfMassMoments, int x, int y, int id)
{
	cv::Mat objectMoment = centerOfMassMoments[id];

	for (int q = 0; q < objectMoment.rows; q++)
	{
		for (int p = 0; p < objectMoment.cols; p++)
		{
			cv::Point centerOfMass = calculateCenterOfMass(coordinateMoments[id]);
			objectMoment.at<float>(q, p) += pow(x - centerOfMass.x, p) * pow(y - centerOfMass.y, q);
		}
	}

	centerOfMassMoments[id] = objectMoment;
}

void FeaturesDetector::processCoordinate(std::vector<cv::Mat>& coordinateMoments, int id, int x, int y)
{
	id--;
	cv::Mat objectMoment = coordinateMoments[id];
	for (int q = 0; q < objectMoment.rows; q++)
	{
		for (int p = 0; p < objectMoment.cols; p++)
		{
			objectMoment.at<float>(q, p) += pow(x, p) * pow(y, q);
		}
	}

	coordinateMoments[id] = objectMoment;
}

void FeaturesDetector::processIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const std::vector<cv::Point>  lookAroundMatrix, int& id)
{
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<float>(y, x) != 0.0f)
			{
				pixels.push_back(cv::Point(x, y));
				assignIndexes(src, dst, pixels, lookAroundMatrix, id);
			}
		}
	}
}

void FeaturesDetector::assignIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const std::vector<cv::Point> lookAroundMatrix, int& id)
{
	bool isIndexed = false;

	while (!pixels.empty())
	{
		cv::Point pixel = pixels.back();
		pixels.pop_back();

		if (checkBoundaries(pixel, src))
		{
			if (dst.at<float>(pixel) != 0.0f)
			{
				continue;
			}

			for (cv::Point neighbor : lookAroundMatrix)
			{
				cv::Point pixelToCheck = getPixelToCheck(pixel, neighbor);

				if (src.at<float>(pixelToCheck) != 0.0f)
				{
					pixels.push_back(pixelToCheck);
				}
			}

			isIndexed = true;
		}
		dst.at<float>(pixel) = id;
	}

	if (isIndexed)
	{
		id++;
	}
}

void FeaturesDetector::assignRandomColors(cv::Mat src, cv::Mat& dst, int count)
{
	std::vector<cv::Vec3f> colors;
	ColorHelper::generateColors(count + 1, colors); // plus background.

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			dst.at<cv::Vec3f>(y, x) = colors[src.at<float>(y, x)];
		}
	}
}

float FeaturesDetector::calculateArea(cv::Mat objectMoment)
{
	return objectMoment.at<float>(0, 0);
}

cv::Point FeaturesDetector::calculateCenterOfMass(cv::Mat coordinateMoment)
{
	cv::Point centerOfMass;
	float m10 = coordinateMoment.at<float>(1, 0);
	float m01 = coordinateMoment.at<float>(0, 1);
	float m00 = calculateArea(coordinateMoment);

	centerOfMass.x = m10 / m00;
	centerOfMass.y = m01 / m00;
	
	return centerOfMass;
}

void FeaturesDetector::calculateMassCenters(std::vector<cv::Mat>& coordinateMoments, std::vector<cv::Point>& massCenters)
{
	for (cv::Mat coordinateMoment : coordinateMoments) {
		massCenters.push_back(calculateCenterOfMass(coordinateMoment));
	}
}

void FeaturesDetector::featureExtraction(cv::Mat& indexedImg, std::vector<cv::Mat>& coordinateMoments, std::vector<float>& circumferenceAreas, int objectsCount, std::vector<cv::Mat>& centerOfMassMoments, std::vector<float>& F1, std::vector<float>& F2, std::vector<float>& uMin, std::vector<float>& uMax, std::vector<float>& coordinateAreas, std::vector<float>& centerOfMassAreas, std::vector<cv::Point>& massCenters)
{
	getCoordinateMoments(indexedImg, coordinateMoments, circumferenceAreas, objectsCount);
	getCenterOfMassMoments(indexedImg, coordinateMoments, centerOfMassMoments);
	getClassificationMoments(centerOfMassMoments, circumferenceAreas, F1, F2, uMin, uMax);
	calculateAreas(coordinateMoments, coordinateAreas);
	calculateAreas(centerOfMassMoments, centerOfMassAreas);
	calculateMassCenters(coordinateMoments, massCenters);
}

void FeaturesDetector::calculateFeatures(float& F1, float& F2, float circumference, float area, float uMin, float uMax)
{
	F1 = pow(circumference, 2) / (100.0f * area);
	F2 = uMin / uMax;
}

void FeaturesDetector::calculateuMinuMax(float& uMin, float& uMax, cv::Mat centerOfTheMassMoment)
{
	float firstPart = 0.5f * (centerOfTheMassMoment.at<float>(0, 2) + centerOfTheMassMoment.at<float>(2, 0));
	float secondPart = 0.5f * sqrt(4 * pow(centerOfTheMassMoment.at<float>(1, 1), 2) + pow(centerOfTheMassMoment.at<float>(0, 2) - centerOfTheMassMoment.at<float>(2, 0), 2));
	uMin = firstPart - secondPart;
	uMax = firstPart + secondPart;
}

bool FeaturesDetector::checkBoundaries(cv::Point pixel, cv::Mat src)
{
	return pixel.x > 0 && pixel.y > 0 && pixel.x <= src.cols && pixel.y <= src.rows;
}

std::vector<cv::Point> FeaturesDetector::etalonsComputing(cv::Mat src, cv::Mat& dst, std::vector<float> F1, std::vector<float> F2, int count)
{
	std::vector<cv::Point> etalon_classes;
	dst = cv::Mat(src.size(), CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));

	// check wrong data
	for (int i = 1; i < count; i++) {
		if (isnan(F1[i]) || isnan(F2[i])) {
			count -= 1;
			F1.erase(F1.begin() + i);
			F2.erase(F2.begin() + i);
		}
	}

	float old_f1 = F1[1];
	float old_f2 = F2[1];
	float etalon_f1 = 0;
	float etalon_f2 = 0;
	int Nr = 0;
	for (int i = 1; i < count; i++) {
		cv::circle(dst, cv::Point(F1[i] * dst.cols, F2[i] * dst.rows), 3, cv::Vec3f(1.0, 1.0, 1.0));
		if (round(F1[i]) == round(old_f1) && round(F2[i]) == round(old_f2))
		{
			etalon_f1 += F1[i];
			etalon_f2 += F2[i];
			Nr += 1;
		}
		else {	// add etalon new class
			cv::Point x;
			x.x = (1.0f / (float)Nr) * etalon_f1;
			x.y = (1.0f / (float)Nr) * etalon_f2;
			etalon_classes.push_back(x);
			etalon_f1 = 0;
			etalon_f2 = 0;
			Nr = 0;
		}
		if (i == count - 1) {	// add etalon last class
			cv::Point x;
			x.x = (1.0f / (float)Nr) * etalon_f1;
			x.y = (1.0f / (float)Nr) * etalon_f2;
			etalon_classes.push_back(x);
		}
		old_f1 = F1[i];
		old_f2 = F2[i];
	}
	for (int i = 0; i < etalon_classes.size(); i++) {
		cv::Point etalon = etalon_classes[i];
		float x = etalon.x * dst.cols;
		float y = etalon.y * dst.rows;
		cv::Vec3f color = ColorHelper::generateColor();
		cv::circle(dst, cv::Point(x, y), 4, color, -1);
	}

	return etalon_classes;
	return std::vector<cv::Point>();
}

void FeaturesDetector::printFeaturesToConsole(std::vector<cv::Point> massCenters, std::vector<float> coordinateAreas, std::vector<float> centerOfMassAreas, std::vector<float> circumferenceAreas, std::vector<float> F1s, std::vector<float> F2s, std::vector<float> uMins, std::vector<float> uMaxes)
{
	for (int i = 0; i < coordinateAreas.size(); i++) 
	{
		std::cout << "id: " << i + 1 << "	";
		std::cout << "center:" << massCenters[i].x << ", " << massCenters[i].y << "	";
		//std::cout << "coordinate area: " << coordinateAreas[i] << " ";
		std::cout << "area: " << centerOfMassAreas[i] << "	";
		std::cout << "circumference: " << circumferenceAreas[i] << " ";
		std::cout << "F1: " << F1s[i] << "	F2: " << F2s[i] << "	";
		std::cout << "uMin: " << uMins[i] << " uMax: " << uMaxes[i] << std::endl;
	}
}

std::string FeaturesDetector::generateText(int id, float value)
{
	std::stringstream stringStream;
	stringStream << id << ": " << value;
	return stringStream.str();
}