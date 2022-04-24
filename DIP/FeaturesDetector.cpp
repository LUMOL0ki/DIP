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

int FeaturesDetector::objectIndexing(cv::Mat src, cv::Mat& dst)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	std::vector<cv::Point> pixels;
	const cv::Point lookAroundMatrix[] = {
		cv::Point(1, 1), cv::Point(1, 0),	cv::Point(1, -1),
		cv::Point(0, 1), /* you are here */	cv::Point(0, -1),
		cv::Point(-1, 1), cv::Point(-1, 0), cv::Point(-1, 1)
	};
	int lookAroundMatrixSize = sizeof(lookAroundMatrix) / sizeof(cv::Point);
	int id = 1;

	proccesingIndexes(src, dst, pixels, lookAroundMatrix, lookAroundMatrixSize, id);

	return id;
}

void FeaturesDetector::proccesingIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const cv::Point  lookAroundMatrix[8], int lookAroundMatrixSize, int& id)
{
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<float>(y, x) != 0.0f)
			{
				pixels.push_back(cv::Point(x, y));
				assignIndexes(src, dst, pixels, lookAroundMatrix, lookAroundMatrixSize, id);
			}
		}
	}
}

void FeaturesDetector::assignIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const cv::Point  lookAroundMatrix[8], int lookAroundMatrixSize, int& id)
{
	bool isIndexed = false;

	while (!pixels.empty())
	{
		cv::Point pixel = pixels.back();
		pixels.pop_back();

		if (pixel.x > 0 && pixel.y > 0 && pixel.x <= src.cols && pixel.y <= src.rows)
		{
			if (dst.at<float>(pixel) != 0.0f)
			{
				continue;
			}

			CheckNeighboringPixels(src, pixels, pixel, lookAroundMatrix, lookAroundMatrixSize);

			isIndexed = true;
		}
		dst.at<float>(pixel) = id;
	}

	if (isIndexed)
	{
		id++;
	}
}

void FeaturesDetector::CheckNeighboringPixels(cv::Mat& src, std::vector<cv::Point>& pixels, cv::Point& currentPixel, const cv::Point lookAroundMatrix[8], int lookAroundMatrixSize)
{
	for (int i = 0; i < lookAroundMatrixSize; i++)
	{
		cv::Point pixelToCheck = currentPixel + lookAroundMatrix[i];
		if (src.at<float>(pixelToCheck) != 0.0f)
		{
			pixels.push_back(pixelToCheck);
		}
	}
}

void FeaturesDetector::assignRandomColors(cv::Mat src, cv::Mat& dst, int count)
{
	std::vector<cv::Vec3f> colors;
	ColorHelper::generateColors(count, colors);

	for (int y = 0; y < src.rows; y++) 
	{
		for (int x = 0; x < src.cols; x++) 
		{
			dst.at<cv::Vec3f>(y, x) = colors[src.at<float>(y, x)];
		}
	}
}

