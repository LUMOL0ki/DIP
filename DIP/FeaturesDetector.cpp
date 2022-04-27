#include "stdafx.h"
#include "FeaturesDetector.h"
#include "colorHelper.h"
#include "matrixHelper.h"
#include <random>

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

	featureExtraction(src1, indexedtrainImg, trainObjects);
	featureExtraction(src2, indexedtestImg, testObjects);
	etalons = etalonsComputing(indexedtrainImg, cl, trainObjects);
	processEtalons(indexedtestImg, dst, testObjects, etalons);
}

void FeaturesDetector::kmeansClustering(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst)
{
	cv::Mat indexedtrainImg = cv::Mat::zeros(src1.size(), src1.type());
	dst = cv::Mat::zeros(src1.size(), CV_32FC3);
	cv::Mat indexedtestImg = cv::Mat::zeros(src1.size(), src1.type());
	dst = cv::Mat::zeros(src1.size(), CV_32FC3);
	std::vector<Object> trainObjects;
	std::vector<Object> testObjects;
	std::vector<cv::Point2f> kmeans;

	featureExtraction(src1, indexedtrainImg, trainObjects);
	featureExtraction(src2, indexedtestImg, testObjects);
	kmeans = kmeansComputing(indexedtrainImg, cl, trainObjects);
	processkmeans(indexedtestImg, dst, testObjects, kmeans);
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

std::vector<cv::Point2f> FeaturesDetector::kmeansComputing(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, int numberOfClusters, int steps)
{
	std::vector<cv::Vec3f> colors;
	dst = cv::Mat(src.size(), CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_int_distribution<> distrx(0, 1);
	std::uniform_int_distribution<> distry(0, 1);

	std::vector<cv::Point2f> centroids;
	std::vector<cv::Point2f> points;
	bool find_new_centroids = true;

	for (int i = 0; i < numberOfClusters; i++) {
		cv::Point2f cent;
		cent.x = distrx(eng);
		cent.y = distry(eng);
		centroids.push_back(cent);
		colors.push_back(ColorHelper::generateColor());
	}

	for (Object object : objects) {
		cv::Point2f point;
		point.x = object.getFirstFeature();
		point.y = object.getSecondFeature();
		points.push_back(point);
		cv::circle(dst, cv::Point(object.getFirstFeature() * dst.cols, object.getSecondFeature() * dst.rows), 3, ColorHelper::White());
	}

	// main loop
	std::vector<int> old_assignment;
	while (find_new_centroids) {
		std::vector<int> points_assignment_idx;
		float kmeansChoose = 0;
		// distance for each point
		for (cv::Point2f point : points) {
			int close_idx = 0;
			float distance = 10000.0f;

			for (cv::Point2f centroid : centroids) {
				float dist;
				cv::Point2f cent = centroid;
				dist = pow(cent.x - point.x, 2) + pow(cent.y - point.y, 2);
				dist = sqrt(dist);	// new distance

				if (dist < distance)
				{
					distance = dist;
					close_idx = kmeansChoose;
				}
				kmeansChoose++;
			}
			points_assignment_idx.push_back(close_idx);
		}

		int i = 0;
		// calculate new centroids
		for (cv::Point2f centroid : centroids) {
			int j = 0;
			int numOfPoints = 0;
			float meanx = 0;
			float meany = 0;
			for (cv::Point2f point : points) 
			{
				if (points_assignment_idx[j] == i)
				{	// if is assigned to centroid
					numOfPoints += 1;
					meanx += point.x;
					meany += point.y;
				}
				j++;
			}
			if (numOfPoints > 0) {
				cv::Point2f cent;
				cent.x = meanx / numOfPoints;
				cent.y = meany / numOfPoints;
				centroid = cent;
			}
			i++;
		}

		if (old_assignment == points_assignment_idx)
			find_new_centroids = false;
		else
			old_assignment.swap(points_assignment_idx);
		steps++;
	}
	// end of loop - final centroids

	for (cv::Point2f centroid : centroids) {
		int i = 0;
		cv::Point2f cent = centroid;
		cv::circle(dst, cv::Point(cent.x * dst.cols, cent.y * dst.rows), 4, colors[i], -1);
		i++;
	}
	return centroids;
}

void FeaturesDetector::processkmeans(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, std::vector<cv::Point2f> kmeans)
{
	dst = src.clone();
	cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
	std::vector<cv::Vec3f> colors;
	ColorHelper::generateColors(kmeans.size(), colors, false);

	for (Object object : objects) {
		float distance = 10000.0f;
		float centroid_index = 0;
		int centroidChoose = 0;
		for (cv::Point2f kmean : kmeans) {
			float x = kmean.x - object.getFirstFeature();
			float y = kmean.y - object.getSecondFeature();
			float dist;
			dist = pow(x, 2) + pow(y, 2);
			dist = sqrt(dist);

			if (dist < distance)
			{
				distance = dist;
				centroid_index = centroidChoose;
			}
		}

		cv::Point2f center = object.getCenterOfMass();
		cv::Vec3f color = colors[centroid_index];
		cv::circle(dst, cv::Point(center.x, center.y), 7, color, -1);
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