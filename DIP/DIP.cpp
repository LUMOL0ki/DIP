// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FeaturesDetector.h"
#include "backprop.h"
#include "slic.h"
#include "HOG.h"

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

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdImg, convertUcharToFloat(threshold));
	featuresDetector.objectIndexingColoredAndNumbered(thresholdImg, indexColoredAndNumberedImg);

	cv::imshow("Features", indexColoredAndNumberedImg);
	cv::waitKey(0); // wait until keypressed
}

void etalonsTask() {
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat testImg = loadImage("images/test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdedTrainImg;
	cv::Mat thresholdedTestImg;
	uchar threshold = 200;
	cv::Mat clusterImg;
	cv::Mat etalonsImg;

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdedTrainImg, convertUcharToFloat(threshold));
	featuresDetector.imageThresholding(testImg, thresholdedTestImg, convertUcharToFloat(0));
	featuresDetector.etalonsClassification(thresholdedTrainImg, thresholdedTestImg, clusterImg, etalonsImg);

	cv::imshow("Cluster", clusterImg);
	cv::imshow("Etalons", etalonsImg);
	cv::waitKey(0); // wait until keypressed
}

void kmeansTask() {
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat testImg = loadImage("images/test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdedTrainImg;
	cv::Mat thresholdedTestImg;
	uchar threshold = 200;
	cv::Mat clusterImg;
	cv::Mat kmeansImg;

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdedTrainImg, convertUcharToFloat(threshold));
	featuresDetector.imageThresholding(testImg, thresholdedTestImg, convertUcharToFloat(0));
	featuresDetector.kmeansClustering(thresholdedTrainImg, thresholdedTestImg, clusterImg, kmeansImg);

	cv::imshow("Cluster", clusterImg);
	cv::imshow("Kmeans", kmeansImg);
	cv::waitKey(0); // wait until keypressed
}

void train(NN* nn)
{
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdedTrainImg;
	uchar threshold = 200;
	std::vector<Object> Objects;

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdedTrainImg, convertUcharToFloat(threshold));
	Objects = featuresDetector.ExteractObjects(thresholdedTrainImg);
	
	int n = Objects.size();
	double** trainingSet = new double* [n];
	
	for (Object object : Objects) 
	{
		int i = object.getId() - 1;
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];
		trainingSet[i][0] = object.getFirstFeature();
		trainingSet[i][1] = object.getSecondFeature();

		switch (object.type)
		{
		case Square:
			trainingSet[i][nn->n[0]] = 1;
			trainingSet[i][nn->n[0] + 1] = 0;
			trainingSet[i][nn->n[0] + 2] = 0;
			break;
		case Star:
			trainingSet[i][nn->n[0]] = 0;
			trainingSet[i][nn->n[0] + 1] = 1;
			trainingSet[i][nn->n[0] + 2] = 0;
			break;
		case Rectangle:
			trainingSet[i][nn->n[0]] = 0;
			trainingSet[i][nn->n[0] + 1] = 0;
			trainingSet[i][nn->n[0] + 2] = 1;
			break;
		default:
			break;
		}
	}

	double error = 1.0;
	int i = 0;

	while (error > 0.01)
	{
		setInput(nn, trainingSet[i % n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (int i = 0; i < n; i++) 
	{
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, int num_samples = 10)
{
	cv::Mat trainImg = loadImage("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat testImg = loadImage("images/test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat thresholdedTrainImg;
	cv::Mat thresholdedTestImg;
	uchar threshold = 200;
	cv::Mat clusterImg;
	cv::Mat etalonsImg;

	FeaturesDetector featuresDetector = FeaturesDetector();
	featuresDetector.imageThresholding(trainImg, thresholdedTrainImg, convertUcharToFloat(threshold));
	featuresDetector.imageThresholding(testImg, thresholdedTestImg, convertUcharToFloat(0));
	std::vector<Object> objects = featuresDetector.etalonsClassification(thresholdedTrainImg, thresholdedTestImg, clusterImg, etalonsImg);

	double* in = new double[nn->n[0]];
	int num_err = 0;

	for (Object object : objects)
	{
		in[0] = object.getFirstFeature();
		in[1] = object.getSecondFeature();

		switch (object.type)
		{
		case Square:
			printf("predicted: Square\n");
			break;
		case Star:
			printf("predicted: Star\n");
			break;
		case Rectangle:
			printf("predicted: Rectangle\n");
			break;
		default:
			break;
		}
		
		setInput(nn, in, true);

		feedforward(nn);

		int output = getOutput(nn, true);

		switch (output)
		{
		case Square:
			printf("It is Square.\n");
			break;
		case Star:
			printf("It is Star.\n");
			break;
		case Rectangle:
			printf("It is Rectangle.\n");
			break;
		default:
			break;
		}

		if (output == object.type) num_err++;
		printf("\n");
	}

	double err = (double)num_err / num_samples;
	printf("test error: %.2f\n", err);
}

void neuralNetworkTask() 
{
	NN* nn = createNN(2, 5, 3);
	train(nn);
	//cv::waitKey(0); // wait until keypressed
	test(nn, 100);
	//cv::waitKey(0); // wait until keypressed
	releaseNN(nn);
}

void HOGTask()
{
	cv::Mat img = loadImage("images/hog_test.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat gradientImage, orientationImage, histograms;

	HOG hog = HOG(img);
	hog.compute(gradientImage, orientationImage);

	cv::imshow("Gradient Image", gradientImage);
	cv::imshow("Orientation Image", orientationImage);

	//hog.createHistograms(gradientImage, orientationImage, 2, 8, 9, histograms);

	//cv::imshow("Histograms", histograms);

	cv::waitKey(0); // wait until keypressed
}

void slic()
{
	cv::Mat src = loadImage("images/slic_bears.jpg", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat dst = src.clone();
	
	cv::imshow("Slic Image: ", src);
	Slic slic = Slic(src, 15, 15, 200);
	slic.computeSlic();
	cv::waitKey(0); // wait until keypressed
}

int main()
{
	//thresholdingTask();
	//indexingTask();
	//featuresFromMomentsTask();
	//etalonsTask();
	//kmeansTask();
	//neuralNetworkTask();
	slic(); // TODO
	//HOGTask(); // TODO 
	return 0;
}