#include "stdafx.h"
#include "HOG.h"

HOG::HOG(cv::Mat src)
{
	this->src = src;
	cvtColor(src, graySrc, CV_BGR2GRAY);
	graySrc.convertTo(graySrc, CV_32F, 1.0 / 255.0);
}

void HOG::compute(cv::Mat& gradientImage, cv::Mat& orientationImage)
{
	cv::Mat gx, gy;

	gx.create(src.size(), CV_32FC1);
	gy.create(src.size(), CV_32FC1);
	gradientImage.create(src.size(), CV_32FC1);
	orientationImage.create(src.size(), CV_32FC1); // = cv::Mat::zeros(src.rows, src.cols, CV_32FC1)
	

	brightnessDifference(src, gx, gy);
	computeSizeOfGradient(src, gx, gy, gradientImage);
	computeOrientationOfGradient(src, gx, gy, orientationImage);
}

void HOG::createHistograms(cv::Mat gradientImage, cv::Mat orientationImage, int blockSize, int cellSize, int numBins, cv::Mat& histograms)
{
    int numCellsX = orientationImage.cols / cellSize;
    int numCellsY = orientationImage.rows / cellSize;
    int numBlocksX = numCellsX - blockSize + 1;
    int numBlocksY = numCellsY - blockSize + 1;

    histograms.create(numBlocksX * numBlocksY, blockSize * blockSize * numBins, CV_32F);

    int cellWidth = cellSize;
    int cellHeight = cellSize;
    int binWidth = 360 / numBins;

    for (int i = 0; i < numBlocksY; i++) {
        for (int j = 0; j < numBlocksX; j++) {
            cv::Mat blockHistogram = histograms.row(i * numBlocksX + j);

            for (int k = i; k < i + blockSize; k++) {
                for (int l = j; l < j + blockSize; l++) {
                    cv::Mat cellOrientation = orientationImage(cv::Rect(l * cellWidth, k * cellHeight, cellWidth, cellHeight));
                    cv::Mat cellGradient = gradientImage(cv::Rect(l * cellWidth, k * cellHeight, cellWidth, cellHeight));

                    for (int m = 0; m < cellHeight; m++) {
                        for (int n = 0; n < cellWidth; n++) {
                            int bin = cellOrientation.at<float>(m, n) / binWidth;
                            float weight = cellGradient.at<float>(m, n);

                            blockHistogram.at<float>(0, (k - i) * blockSize * numBins + (l - j) * numBins + bin) += weight;
                        }
                    }
                }
            }

            cv::normalize(blockHistogram, blockHistogram, 1, 0, cv::NORM_L1);
        }
    }
}

void HOG::computeOrientationOfGradient(cv::Mat src, cv::Mat gx, cv::Mat gy, cv::Mat& orientationImage)
{
    for (int y = 0; y < src.rows; y++) 
	{
        for (int x = 0; x < src.cols; x++) 
		{
            float angle = atan2(gy.at<float>(y, x), gx.at<float>(y, x)) * 180 / CV_PI;

            if (angle < 0) 
			{
                angle += 180;
            }

            orientationImage.at<float>(y, x) = angle;
        }
    }
}

void HOG::computeSizeOfGradient(cv::Mat src, cv::Mat gx, cv::Mat gy, cv::Mat& gradientImage)
{
	for (int y = 0; y < gx.rows; y++) 
	{
		for (int x = 0; x < gx.cols; x++) 
		{
			double dx = gx.at<float>(y, x);
			double dy = gy.at<float>(y, x);
			gradientImage.at<float>(y, x) = sqrt(pow(gx.at<float>(y, x), 2) + pow(gy.at<float>(y, x), 2));
		}
	}
}

void HOG::brightnessDifference(cv::Mat src, cv::Mat& gx, cv::Mat& gy)
{
	for (int y = 0; y < src.rows - 1; y++)
	{
		for (int x = 0; x < src.cols - 1; x++)
		{
			float fx = src.at<float>(y, x + 1) - src.at<float>(y, x);
			float fy = src.at<float>(y + 1, x) - src.at<float>(y, x);

			gx.at<float>(y, x) = fx;
			gy.at<float>(y, x) = fy;
		}
	}
}
