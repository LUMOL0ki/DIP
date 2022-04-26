// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FeaturesDetector.h"

void convertColorToGray32(cv::Mat src, cv::Mat& dst) {
	cv::cvtColor(src.clone(), dst, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	dst.convertTo(dst, CV_32FC3, 1.0 / 255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0
}

float convertUcharToFloat(uchar input) {
	return (float)input / 255;
}

cv::Mat loadImage(const cv::String &filename, int flags = 1) {
	cv::Mat src_8uc3_img = cv::imread(filename, flags); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)

	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	return src_8uc3_img;
}

void thresholdingTask() {
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdImg;
	uchar threshold = 200;

	convertColorToGray32(trainImg, trainImg);
		
	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdImg, convertUcharToFloat(threshold));

	cv::imshow("Thresholding", thresholdImg);
	cv::waitKey(0); // wait until keypressed
}

void indexingTask() {
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdImg;
	uchar threshold = 204;
	cv::Mat indexColoredImg;

	convertColorToGray32(trainImg, trainImg);

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdImg, convertUcharToFloat(threshold));
	featuresDetector.objectIndexingColored(thresholdImg, indexColoredImg);

	cv::imshow("Indexing", indexColoredImg);
	cv::waitKey(0); // wait until keypressed
}

void featuresFromMomentsTask() {
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdImg;
	uchar threshold = 200;
	cv::Mat indexColoredAndNumberedImg;

	convertColorToGray32(trainImg, trainImg);

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdImg, convertUcharToFloat(threshold));
	featuresDetector.objectIndexingColoredAndNumbered(thresholdImg, indexColoredAndNumberedImg);

	cv::imshow("Features", indexColoredAndNumberedImg);
	cv::waitKey(0); // wait until keypressed
}

void etalonsTask() {
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat testImg = loadImage("images/test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdImg;
	uchar threshold = 240;
	cv::Mat clusterImg;
	cv::Mat etalonsImg;

	convertColorToGray32(trainImg, trainImg);

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdImg, convertUcharToFloat(threshold));
	featuresDetector.etalonsClassification(trainImg, testImg, clusterImg, etalonsImg);

	cv::imshow("Cluster", clusterImg);
	cv::imshow("Etalons", etalonsImg);
	cv::waitKey(0); // wait until keypressed
}

void kmeansTask() {
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat testImg = loadImage("images/test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdImg;
	uchar threshold = 200;
	cv::Mat kmeansImg;

	convertColorToGray32(trainImg, trainImg);

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdImg, convertUcharToFloat(threshold));
	//featuresDetector.kmeansClustering(trainImg, testImg, etalonsImg);

	cv::imshow("Etalons", kmeansImg);
	cv::waitKey(0); // wait until keypressed
}

int main()
{
	//starter();
	//thresholdingTask();
	//indexingTask();
	//featuresFromMomentsTask();
	etalonsTask();
	//kmeansTask();
	return 0;
}