#include "stdafx.h"
#include "FeaturesDetector.h"
#include "colorHelper.h"
#include "matrixHelper.h"

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

	int objectsCount = Object::getNumberOfObjects(src, indexedImg);
	assignRandomColors(indexedImg, dst, objectsCount);
}

void FeaturesDetector::objectIndexingColoredAndNumbered(cv::Mat src, cv::Mat& dst)
{
	cv::Mat indexedImg = cv::Mat::zeros(src.size(), src.type());
	dst = cv::Mat::zeros(src.size(), CV_32FC3);
	std::vector<Object> objects;

	int objectsCount = featureExtraction(src, indexedImg, objects);
	assignRandomColors(indexedImg, dst, objectsCount);
	AssignText(objects, dst);
	printFeaturesToConsole(objects);
}

void FeaturesDetector::AssignText(std::vector<Object> objects, cv::Mat& dst)
{
	for (Object object : objects) {
		cv::Point centerOfMass = object.getCenterOfMass();
		cv::putText(dst, generateText(object.getId(), object.getCenterOfMassArea()), cv::Point(centerOfMass.x - 12, centerOfMass.y), cv::FONT_HERSHEY_SIMPLEX, 0.3, ColorHelper::White());
		cv::putText(dst, generateText(object.getId(), object.getCircumferenceArea()), cv::Point(centerOfMass.x - 12, centerOfMass.y + 11), cv::FONT_HERSHEY_SIMPLEX, 0.3, ColorHelper::White());
	}
}

void FeaturesDetector::etalonsClassification(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst)
{
	cv::Mat indexedtrainImg = cv::Mat::zeros(src1.size(), src1.type());
	dst = cv::Mat::zeros(src1.size(), CV_32FC3);
	cv::Mat indexedtestImg = cv::Mat::zeros(src1.size(), src1.type());
	dst = cv::Mat::zeros(src1.size(), CV_32FC3);
	std::vector<Object> trainObjects;
	std::vector<Object> testObjects;
	std::vector<cv::Point2f> etalons;

	int objectsTrainCount = featureExtraction(src1, indexedtrainImg, trainObjects);
	int objectsTestCount = featureExtraction(src2, indexedtestImg, testObjects);
	etalons = etalonsComputing(indexedtrainImg, cl, trainObjects);
	processEtalons(indexedtestImg, dst, testObjects, etalons);
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

int FeaturesDetector::featureExtraction(cv::Mat src, cv::Mat& dst, std::vector<Object>& objects)
{
	int objectsCount = Object::getNumberOfObjects(src, dst);

	for (int i = 0; i < objectsCount; i++)
	{
		objects.push_back(Object(i + 1, dst));
	}

	return objectsCount;
}

std::vector<cv::Point2f> FeaturesDetector::etalonsComputing(cv::Mat src, cv::Mat& dst, std::vector<Object> objects)
{
	std::vector<cv::Point2f> etalons;
	dst = cv::Mat(src.size(), CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
	float old_f1 = objects[0].getFirstFeature();
	float old_f2 = objects[0].getSecondFeature();
	float etalon_f1 = 0;
	float etalon_f2 = 0;
	int Nr = 0;

	for (Object object : objects) {
		cv::circle(dst, cv::Point2f(object.getFirstFeature() * dst.cols, object.getSecondFeature() * dst.rows), 3, ColorHelper::White());
		
		if (round(object.getFirstFeature()) == round(old_f1) && round(object.getSecondFeature()) == round(old_f2))
		{
			etalon_f1 += object.getFirstFeature();
			etalon_f2 += object.getSecondFeature();
			Nr += 1;
		}
		else 
		{	// add etalon new class
			cv::Point2f x;
			x.x = (1.0f / (float)Nr) * etalon_f1;
			x.y = (1.0f / (float)Nr) * etalon_f2;
			etalons.push_back(x);
			etalon_f1 = 0;
			etalon_f2 = 0;
			Nr = 0;
		}

		if (object.getId() == objects.size()) 
		{	
			cv::Point2f x;
			x.x = (1.0f / (float)Nr) * etalon_f1;
			x.y = (1.0f / (float)Nr) * etalon_f2;
			etalons.push_back(x);
		}

		old_f1 = object.getFirstFeature();
		old_f2 = object.getSecondFeature();
	}

	for(cv::Point2f etalon : etalons) 
	{
		float x = etalon.x * dst.cols;
		float y = etalon.y * dst.rows;
		cv::Vec3f color = ColorHelper::generateColor();
		cv::circle(dst, cv::Point2f(x, y), 4, color, -1);
	}

	return etalons;
}

void FeaturesDetector::processEtalons(cv::Mat src, cv::Mat& dst, std::vector<Object> testObjects, std::vector<cv::Point2f> etalons)
{
	dst = src.clone();
	cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
	std::vector<cv::Vec3f> colors;
	ColorHelper::generateColors(etalons.size(), colors, false);

	for (int j = 0; j < etalons.size(); j++) {
		colors.push_back(ColorHelper::generateColor());
	}

	for (Object object : testObjects) {
		float distance = 100.0f;
		float etalon_index = -1;
		float etalonChoose = 0;
		for (cv::Point2f etalon : etalons) {
			float x = etalon.x - object.getFirstFeature();
			float y = etalon.y - object.getSecondFeature();
			float dist;
			dist = sqrt(pow(x, 2) + pow(y, 2));

			if (dist < distance)
			{
				distance = dist;
				etalon_index = etalonChoose;
			}
			etalonChoose++;
		}

		cv::Point2f center = object.getCenterOfMass();
		cv::Vec3f color = colors[etalon_index];
		cv::circle(dst, object.getCenterOfMass(), 7, color, -1);
	}
}

void FeaturesDetector::printFeaturesToConsole(std::vector<Object> objects)
{
	for (Object object : objects)
	{
		std::cout << "id: " << object.getId() << "	";
		std::cout << "center:" << object.getCenterOfMass().x << ", " << object.getCenterOfMass().y << "	";
		//std::cout << "coordinate area: " << coordinateAreas[i] << " ";
		std::cout << "area: " << object.getCenterOfMassArea() << "	";
		std::cout << "circumference: " << object.getCircumferenceArea() << " ";
		std::cout << "F1: " << object.getFirstFeature() << "	F2: " << object.getSecondFeature() << "	";
		Boundary boundary = object.getBoundary();
		std::cout << "uMin: " << boundary.uMin << " uMax: " << boundary.uMax << std::endl;
	}
}

std::string FeaturesDetector::generateText(int id, float value)
{
	std::stringstream stringStream;
	stringStream << id << ": " << value;
	return stringStream.str();
}