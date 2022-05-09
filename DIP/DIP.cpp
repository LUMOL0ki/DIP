// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FeaturesDetector.h"
#include "backprop.h"
#include "slic.h"

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
	cv::Mat thresholdedTrainImg;
	cv::Mat thresholdedTestImg;
	uchar threshold = 200;
	cv::Mat clusterImg;
	cv::Mat etalonsImg;

	convertColorToGray32(trainImg, trainImg);
	convertColorToGray32(testImg, testImg);

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

	convertColorToGray32(trainImg, trainImg);
	convertColorToGray32(testImg, testImg);

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
	int n = 1000;
	double** trainingSet = new double* [n];
	for (int i = 0; i < n; i++) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

		bool classA = i % 2;

		for (int j = 0; j < nn->n[0]; j++) {
			if (classA) {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}

		trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
		trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;
	}

	double error = 1.0;
	int i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i % n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (int i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, int num_samples = 10)
{
	double* in = new double[nn->n[0]];

	int num_err = 0;
	for (int n = 0; n < num_samples; n++)
	{
		bool classA = rand() % 2;

		for (int j = 0; j < nn->n[0]; j++)
		{
			if (classA)
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}
		printf("predicted: %d\n", !classA);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classA) num_err++;
		printf("\n");
	}
	double err = (double)num_err / num_samples;
	printf("test error: %.2f\n", err);
}

void neuralNetworkTask() 
{
	NN* nn = createNN(2, 4, 2);
	train(nn);
	cv::waitKey(0); // wait until keypressed
	test(nn, 100);
	cv::waitKey(0); // wait until keypressed
	releaseNN(nn);
}

void slic()
{
	cv::Mat src = loadImage("images/slic_bears.jpg", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	cv::Mat dst = src.clone();
	
	cv::cvtColor(dst, src, CV_BGR2Lab);

	double steps = Slic::getNumberOfSteps(src.rows, src.cols, 10);

	Slic slic = Slic(src, steps, 2);
	slic.drawContours(dst, CV_RGB(255, 0, 0));
	cv::imshow("Slic", dst);
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
	slic();
	return 0;
}