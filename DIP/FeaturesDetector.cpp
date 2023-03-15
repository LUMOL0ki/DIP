#include "stdafx.h"
#include "FeaturesDetector.h"
#include "colorHelper.h"
#include "matrixHelper.h"
#include <random>

void FeaturesDetector::imageThresholding(cv::Mat src, cv::Mat& dst, float threshold)
{
	convertColorToGray32(src, src);

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

void FeaturesDetector::convertColorToGray32(cv::Mat src, cv::Mat& dst)
{
	cv::cvtColor(src.clone(), dst, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	dst.convertTo(dst, CV_32FC3, 1.0 / 255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0
}

std::vector<Object> FeaturesDetector::etalonsClassification(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst)
{
	cv::Mat indexedtrainImg = cv::Mat::zeros(src1.size(), src1.type());
	cl = cv::Mat::zeros(src1.size(), CV_32FC3);
	cv::Mat indexedtestImg = cv::Mat::zeros(src2.size(), src2.type());
	dst = cv::Mat::zeros(src2.size(), CV_32FC3);
	std::vector<Object> trainObjects;
	std::vector<Object> testObjects;
	std::vector<cv::Point2f> etalons;

	featureExtraction(src1, indexedtrainImg, trainObjects);
	featureExtraction(src2, indexedtestImg, testObjects);
	etalons = etalonsComputing(indexedtrainImg, cl, trainObjects);
	processEtalons(indexedtestImg, dst, testObjects, etalons);
	return testObjects;
}

void FeaturesDetector::kmeansClustering(cv::Mat src1, cv::Mat src2, cv::Mat& cl, cv::Mat& dst)
{
	cv::Mat indexedTrainImg = cv::Mat::zeros(src1.size(), src1.type());
	cl = cv::Mat::zeros(src1.size(), CV_32FC3);
	cv::Mat indexedTestImg = cv::Mat::zeros(src2.size(), src2.type());
	dst = cv::Mat::zeros(src2.size(), CV_32FC3);
	std::vector<Object> trainObjects;
	std::vector<Object> testObjects;
	std::vector<cv::Point2f> kmeans;
	int steps = 0;

	featureExtraction(src1, indexedTrainImg, trainObjects);
	featureExtraction(src2, indexedTestImg, testObjects);
	kmeans = kmeansComputing(indexedTrainImg, cl, trainObjects, steps);
	processkmeans(indexedTestImg, dst, testObjects, kmeans);
}

std::vector<Object> FeaturesDetector::ExteractObjects(cv::Mat src)
{
	cv::Mat indexedImg = cv::Mat::zeros(src.size(), CV_32FC3);
	cv::Mat cl = cv::Mat::zeros(src.size(), CV_32FC3);
	std::vector<Object> Objects;

	featureExtraction(src, indexedImg, Objects);
	etalonsComputing(indexedImg, cl, Objects);
	return Objects;
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

std::vector<cv::Point2f> FeaturesDetector::etalonsComputing(cv::Mat src, cv::Mat& dst, std::vector<Object>& objects)
{
	dst = cv::Mat(src.size(), CV_32FC3, ColorHelper::Black());
	std::vector<cv::Point2f> etalons;
	float currentF1 = objects[0].getFirstFeature();
	float currentF2 = objects[0].getSecondFeature();
	float etalonF1 = 0;
	float etalonF2 = 0;
	int Nr = 0;

	for (Object object : objects) {
		cv::circle(dst, cv::Point2f(object.getFirstFeature() * dst.cols, object.getSecondFeature() * dst.rows), 3, ColorHelper::White());
		
		if (round(object.getFirstFeature()) == round(currentF1) && round(object.getSecondFeature()) == round(currentF2))
		{
			etalonF1 += object.getFirstFeature();
			etalonF2 += object.getSecondFeature();
			Nr++;
		}
		else 
		{	
			cv::Point2f newEtalon;

			calculateEtalon(newEtalon, Nr, etalonF1, etalonF2);
			etalons.push_back(newEtalon);
			
			etalonF1 = 0;
			etalonF2 = 0;
			Nr = 0;
		}

		if (object.getId() == objects.size()) 
		{	
			cv::Point2f newEtalon;
			
			calculateEtalon(newEtalon, Nr, etalonF1, etalonF2);
			etalons.push_back(newEtalon);
		}

		currentF1 = object.getFirstFeature();
		currentF2 = object.getSecondFeature();
	}

	drawEtalons(etalons, dst);

	for (Object& object : objects)
	{
		float currentDistance = 100.0f;
		int etalonId = -1;

		assignIds(etalons, cv::Point2f(object.getFirstFeature(), object.getSecondFeature()), currentDistance, etalonId);
		object.type = static_cast<ObjectType>(etalonId);
	}

	return etalons;
}

void FeaturesDetector::calculateEtalon(cv::Point2f& etalon, int Nr, float etalonF1, float etalonF2)
{
	etalon.x = (1.0f / (float)Nr) * etalonF1;
	etalon.y = (1.0f / (float)Nr) * etalonF2;
}

void FeaturesDetector::processEtalons(cv::Mat src, cv::Mat& dst, std::vector<Object>& objects, std::vector<cv::Point2f> etalons)
{
	dst = src.clone();
	cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
	std::vector<cv::Vec3f> colors;

	ColorHelper::generateColors(etalons.size(), colors, false);

	for (Object& object : objects)
	{
		float currentDistance = 100.0f;
		int etalonId = -1;

		assignIds(etalons, cv::Point2f(object.getFirstFeature(), object.getSecondFeature()), currentDistance, etalonId);
		object.type = static_cast<ObjectType>(etalonId);

		drawCenterOfObject(object, dst, colors[etalonId]);
	}
}

void FeaturesDetector::assignIds(std::vector<cv::Point2f> srcClasses, cv::Point2f pixel, float& currentDistance, int& currentId)
{
	int newId = 0;

	for (cv::Point2f srcClass : srcClasses)
	{
		float newDistance = calculateDistance(srcClass.x - pixel.x, srcClass.y - pixel.y);

		checkDistance(newDistance, currentDistance, newId, currentId);

		newId++;
	}
}

double FeaturesDetector::calculateDistance(float x, float y)
{
	return sqrt(pow(x, 2) + pow(y, 2));
}

std::vector<cv::Point2f> FeaturesDetector::kmeansComputing(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, int& steps, int numberOfClusters)
{
	dst = cv::Mat(src.size(), CV_32FC3, ColorHelper::Black());

	std::vector<cv::Point2f> centroids;
	std::vector<cv::Point2f> pixels;
	std::vector<int> kmeansIds;
	bool isNewCentroid = true;

	initializeCentroids(numberOfClusters, centroids);
	initializePixels(objects, pixels, dst);

	while (isNewCentroid) 
	{
		std::vector<int> kmeansIds;

		for (cv::Point2f pixel : pixels) {
			float currentDistance = 10000.0f;
			int kmeansId = -1;

			assignIds(centroids, pixel, currentDistance, kmeansId);

			kmeansIds.push_back(kmeansId);
		}

		int i = 0;
		// calculate new centroids
		for (cv::Point2f& centroid : centroids) 
		{
			int numberOfPixels = 0;
			float meanX = 0;
			float meanY = 0;

			int j = 0;
			for (cv::Point2f pixel : pixels) 
			{
				if (kmeansIds[j] == i)
				{	
					numberOfPixels++;
					meanX += pixel.x;
					meanY += pixel.y;
				}
				j++;
			}

			recalculateCentroid(numberOfPixels, centroid, meanX, meanY);
			i++;
		}

		assignIds(kmeansIds, kmeansIds, isNewCentroid);

		steps++;
	}
	// end of loop - final centroids

	drawCentroids(centroids, dst);
	return centroids;
}

void FeaturesDetector::recalculateCentroid(int numOfPixels, cv::Point2f& centroid, float meanX, float meanY)
{
	if (numOfPixels > 0)
	{
		centroid.x = meanX / numOfPixels;
		centroid.y = meanY / numOfPixels;
	}
}

void FeaturesDetector::assignIds(std::vector<int>& kmeansIds, std::vector<int>& newkmeansIds, bool& isNewCentroid)
{
	if (kmeansIds == newkmeansIds)
	{
		isNewCentroid = false;
	}
	else
	{
		kmeansIds.swap(newkmeansIds);
	}
}

void FeaturesDetector::checkDistance(float srcDistance, float& dstDistance, float newId, int& currentId)
{
	if (srcDistance < dstDistance)
	{
		dstDistance = srcDistance;
		currentId = newId;
	}
}

void FeaturesDetector::initializePixels(std::vector<Object>& objects, std::vector<cv::Point2f>& pixels, cv::Mat& dst)
{
	for (Object object : objects)
	{
		cv::Point2f newPixel;
		newPixel.x = object.getFirstFeature();
		newPixel.y = object.getSecondFeature();
		pixels.push_back(newPixel);
		cv::circle(dst, cv::Point(newPixel.x * dst.cols, newPixel.y * dst.rows), 3, ColorHelper::White());
	}
}

void FeaturesDetector::processkmeans(cv::Mat src, cv::Mat& dst, std::vector<Object> objects, std::vector<cv::Point2f> kmeans)
{
	dst = src.clone();
	cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
	std::vector<cv::Vec3f> colors;

	ColorHelper::generateColors(kmeans.size(), colors, false);

	for (Object object : objects) 
	{
		float currentDistance = 10000.0f;
		int centroidId = 0;
		int newCentroidId = 0;
		for (cv::Point2f kmean : kmeans) 
		{
			float newDistance = calculateDistance(kmean.x - object.getFirstFeature(), kmean.y - object.getSecondFeature());
			checkDistance(newDistance, currentDistance, newCentroidId, centroidId);

			newCentroidId++;
		}

		drawCenterOfObject(object, dst, colors[centroidId]);
	}
}

void FeaturesDetector::initializeCentroids(int numberOfClusters, std::vector<cv::Point2f>& centroids)
{
	std::random_device randomDevice;
	std::mt19937 engine(randomDevice());
	std::uniform_int_distribution<> distributionX(0, 1);
	std::uniform_int_distribution<> distributionY(0, 1);

	for (int i = 0; i < numberOfClusters; i++) {
		cv::Point2f newCentroid;
		newCentroid.x = distributionX(engine);
		newCentroid.y = distributionY(engine);
		centroids.push_back(newCentroid);
	}
}

void FeaturesDetector::drawCentroids(std::vector<cv::Point2f> centroids, cv::Mat& dst)
{
	std::vector<cv::Vec3f> colors;
	int colorId = 0;

	ColorHelper::generateColors(centroids.size(), colors, false);

	for (cv::Point2f centroid : centroids)
	{
		cv::circle(dst, cv::Point(centroid.x * dst.cols, centroid.y * dst.rows), 4, colors[colorId], -1);
		colorId++;
	}
}

void FeaturesDetector::drawEtalons(std::vector<cv::Point2f>& etalons, cv::Mat& dst)
{
	for (cv::Point2f etalon : etalons)
	{
		float x = etalon.x * dst.cols;
		float y = etalon.y * dst.rows;
		cv::Vec3f color = ColorHelper::generateColor();
		cv::circle(dst, cv::Point2f(x, y), 4, color, -1);
	}
}

void FeaturesDetector::drawCenterOfObject(Object& object, cv::Mat& dst, cv::Vec3f color)
{
	cv::Point2f center = object.getCenterOfMass();
	cv::circle(dst, cv::Point(center.x, center.y), 7, color, -1);
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