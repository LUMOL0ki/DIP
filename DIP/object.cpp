#include "stdafx.h"
#include "object.h"
#include "matrixHelper.h"


Object::Object(int id, cv::Mat src)
{
	this->id = id;

	coordinateMoment = cv::Mat::zeros(cv::Size(2, 2), src.type());
	centerOfMassMoment = cv::Mat::zeros(cv::Size(3, 3), src.type());
	circumferenceArea = 0.0f;

	initializeCoordinate(src);
	initializeCenterOfMass(src);
	initializeCircumference();
}

bool Object::checkBoundaries(cv::Point pixel, cv::Mat src)
{
	return pixel.x > 0 && pixel.y > 0 && pixel.x <= src.cols && pixel.y <= src.rows;
}

const cv::Point Object::getPixelToCheck(cv::Point pixel, cv::Point neighbor)
{
	return pixel + neighbor;
}

float Object::getAreaFromMoment(cv::Mat objectMoment)
{
	return objectMoment.at<float>(0, 0);
}

void Object::initializeCoordinate(cv::Mat src)
{
	std::vector<cv::Point> lookAroundMatrix = MatrixHelper::getLookAroundMatrix(false);
	int lookAroundMatrixSize = (lookAroundMatrix.size() * sizeof(cv::Point)) / sizeof(cv::Point);

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int detectedId = src.at<float>(y, x);

			if (detectedId == id)
			{
				processCoordinate(x, y);

				cv::Point pixel = cv::Point(x, y);

				if (checkBoundaries(pixel, src))
				{
					int circumferenceCounter = 0;

					for (cv::Point neighbor : lookAroundMatrix)
					{
						if (src.at<float>(getPixelToCheck(pixel, neighbor)) == detectedId)
						{
							circumferenceCounter++;
						}
					}

					if (circumferenceCounter != lookAroundMatrixSize)
					{
						circumferenceArea++;
					}
				}
			}
		}
	}

	coordinateArea = getAreaFromMoment(coordinateMoment);
}

void Object::initializeCenterOfMass(cv::Mat src)
{
	calculateCenterOfMass();

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int detectedId = src.at<float>(y, x);

			if (detectedId == id)
			{
				processCenterOfMass(x, y);
			}
		}
	}

	centerOfMassArea = getAreaFromMoment(centerOfMassMoment);
}

void Object::initializeCircumference()
{
	calculateuMinuMax();
	calculateFeatures();
}

void Object::processCoordinate(int x, int y)
{
	for (int q = 0; q < coordinateMoment.rows; q++)
	{
		for (int p = 0; p < coordinateMoment.cols; p++)
		{
			coordinateMoment.at<float>(q, p) += pow(x, p) * pow(y, q);
		}
	}
}

void Object::processCenterOfMass(int x, int y)
{
	for (int q = 0; q < centerOfMassMoment.rows; q++)
	{
		for (int p = 0; p < centerOfMassMoment.cols; p++)
		{
			centerOfMassMoment.at<float>(q, p) += pow(x - centerOfMass.x, p) * pow(y - centerOfMass.y, q);
		}
	}
}

void Object::calculateCenterOfMass()
{
	float m10 = coordinateMoment.at<float>(1, 0);
	float m01 = coordinateMoment.at<float>(0, 1);
	float m00 = coordinateArea;

	centerOfMass.x = m01 / m00;
	centerOfMass.y = m10 / m00;
}

void Object::calculateuMinuMax()
{
	float u11 = centerOfMassMoment.at<float>(1, 1);
	float u02 = centerOfMassMoment.at<float>(0, 2);
	float u20 = centerOfMassMoment.at<float>(2, 0);
	float firstPart = 0.5f * (u02 + u20);
	float secondPart = 0.5f * sqrt(4 * pow(u11, 2) + pow(u02 - u20, 2));
	boundary.uMin = firstPart - secondPart;
	boundary.uMax = firstPart + secondPart;
}

void Object::calculateFeatures()
{
	F1 = pow(circumferenceArea, 2) / (100.0f * centerOfMassArea);
	F2 = boundary.uMin / boundary.uMax;
}

int Object::getId()
{
	return id;
}

cv::Mat Object::getCoordinateMoment()
{
	return coordinateMoment;
}

cv::Mat Object::getCenterOfMassMoment()
{
	return centerOfMassMoment;
}

cv::Point2f Object::getCenterOfMass()
{
	return centerOfMass;
}

float Object::getCoordinateArea()
{
	return coordinateArea;
}

float Object::getCenterOfMassArea()
{
	return centerOfMassArea;
}

float Object::getCircumferenceArea()
{
	return circumferenceArea;
}

float Object::getFirstFeature()
{
	return F1;
}

float Object::getSecondFeature()
{
	return F2;
}

Boundary Object::getBoundary()
{
	return boundary;
}

int Object::getNumberOfObjects(cv::Mat src, cv::Mat& dst)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	int objectCount = 1;

	initializeObjectsCounting(src, dst, objectCount);

	return objectCount - 1; // minus backgroung;
}

void Object::initializeObjectsCounting(cv::Mat& src, cv::Mat& dst, int& id)
{
	std::vector<cv::Point> pixels;
	std::vector<cv::Point> lookAroundMatrix = MatrixHelper::getLookAroundMatrix();

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<float>(y, x) != 0.0f)
			{
				pixels.push_back(cv::Point(x, y));
				lookForObject(src, dst, pixels, lookAroundMatrix, id);
			}
		}
	}
}

void Object::lookForObject(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const std::vector<cv::Point> lookAroundMatrix, int& id)
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
