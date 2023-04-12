#include "stdafx.h"
#include "slic.h"

Slic::Slic(cv::Mat src, int numberOfSegments, int iterations)
{
    this->src = src;
    this->numberOfSegments = numberOfSegments;
    std::vector<ClusterCenter> clusterCenters = std::vector<ClusterCenter>();
}

void Slic::computeSlic()
{
	int imagePixelSize = src.rows * src.cols;
	printf("%d ", imagePixelSize);
	float step = 0.0f;
	if (numberOfSegments != 0) {
		step = sqrt(imagePixelSize / numberOfSegments);
	}
	cv::Mat input_circles1 = src.clone();
	cv::Mat input_circles2 = src.clone();
	cv::Mat indexer = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	printf("\n Step: %f ,", step);
	int S = static_cast<int>(std::round(step));
	printf("\n S: %d", S);
	// Step 1 visualize centers
	cv::Vec3b red = { 0, 0, 255 }; // because it is BGR
	printf("\n Centers at Regular intervals:");
	int numberCounter = 0;
	// 1a. Initialize cluster centers Ck = [Rk, Gk, Bk, xk, yk] by sampling pixels at regular grid steps S
	for (int rows = static_cast<int>(std::round(S / 2)); rows < src.rows; rows += S) {
		for (int cols = static_cast<int>(std::round(S / 2) - 10); cols < src.cols; cols += S) {
			cv::circle(input_circles1, cv::Point(cols, rows), 3, red, -1);
			cv::Vec3b pixel = src.at<cv::Vec3b>(rows, cols);
			ClusterCenter centers;
			centers.R = pixel[2];
			centers.G = pixel[1];
			centers.B = pixel[0];
			centers.x = cols;
			centers.y = rows;
			clusterCenters.push_back(centers);


			printf("\n %d - centers.R: %d, centers.G: %d, centers.B: %d, centers.x: %d, centers.y. %d ", numberCounter, centers.R, centers.G, centers.B, centers.x, centers.y);
			numberCounter++;
		}
	}
	cv::imshow("Slic Image: dotted ", input_circles1);
	printf("\n Centers at Lowest Gradients:");
	// 1b - move the cluster centers to the lowest gradient position in a 3 × 3	neighborhood.
	for (int i = 0; i < clusterCenters.size(); i++) {
		cv::Point lowestGrad = findLowestGradientPosition(src, clusterCenters[i].x, clusterCenters[i].y);
		cv::Vec3b pixel = src.at<cv::Vec3b>(lowestGrad.y, lowestGrad.x);
		cv::circle(input_circles2, cv::Point(lowestGrad.x, lowestGrad.y), 3, red, -1);
		ClusterCenter centers;
		centers.R = pixel[2];
		centers.G = pixel[1];
		centers.B = pixel[0];
		centers.x = lowestGrad.x;
		centers.y = lowestGrad.y;

		printf("\n %d - centers.R: %d, centers.G: %d, centers.B: %d, centers.x: %d, centers.y. %d ", i, centers.R, centers.G, centers.B, centers.x, centers.y);
	}
	cv::imshow("Slic Image: Lowest Grad ", input_circles2);

	bool stop = true;
	int iterationsCounter = 0;
	float maxDistanceMoved;

	const int maxIterations = 5;
	const float minDistanceThreshold = 2.0;
	do {

		for (int i = 0; i < clusterCenters.size(); i++) {

			int start_x = 0;
			int end_x = 0;
			int start_y = 0;
			int end_y = 0;
			getBoundaries(clusterCenters[i].x, clusterCenters[i].y, src.rows, src.cols, S, start_x, end_x, start_y, end_y);

			int currentClusterX = clusterCenters[i].x;
			int currentClusterY = clusterCenters[i].y;

			for (int y = start_y; y < end_y; y++) {
				for (int x = start_x; x < end_x; x++) {
					cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

					float min_distance = DBL_MAX;
					int closest_cluster = -1;

					for (int c = 0; c < clusterCenters.size(); c++) {
						int R_k = clusterCenters[c].R;
						int R_i = pixel[2];
						int G_k = clusterCenters[c].G;
						int G_i = pixel[1];
						int B_k = clusterCenters[c].B;
						int B_i = pixel[0];
						int m = 10; // You can adjust this value depending on your application
						int x_k = clusterCenters[c].x;
						int x_i = x;
						int y_k = clusterCenters[c].y;
						int y_i = y;

						float d_RGB = sqrt(pow(R_k - R_i, 2) + pow(G_k - G_i, 2) + pow(B_k - B_i, 2));
						float d_xy = sqrt((x_k - x_i) * (x_k - x_i) + (y_k - y_i) * (y_k - y_i));

						float D_s = d_RGB + (((float)m / S) * d_xy);



						float euclideanDistance = sqrt(pow(currentClusterX - clusterCenters[c].x, 2) + pow(currentClusterY - clusterCenters[c].y, 2));
						float pythagoras = sqrt(S * S + S * S);

						if (D_s < min_distance) {
							if (euclideanDistance <= pythagoras) {
								min_distance = D_s;
								closest_cluster = c;
							}
						}


					}

					indexer.at<uchar>(y, x) = static_cast<uchar>(closest_cluster);
				}
			}


		}
		iterationsCounter++;
		maxDistanceMoved = recalculateClusters(src, indexer, clusterCenters);
		cv::Mat display = src.clone();
		printf(" \n -------------------------------------- \n Iteration %d", iterationsCounter);
		for (int i = 0; i < clusterCenters.size(); i++) {
			cv::Point location_xy(clusterCenters[i].x, clusterCenters[i].y);
			cv::Vec3b blue = { 255, 0, 0 };
			cv::circle(display, cv::Point(location_xy.x, location_xy.y), 3, blue, -1);

			printf("\n %d - centers.R: %d, centers.G: %d, centers.B: %d, centers.x: %d, centers.y. %d ", i, clusterCenters[i].R, clusterCenters[i].G, clusterCenters[i].B, clusterCenters[i].x, clusterCenters[i].y);

			int number = i;
			std::string text = std::to_string(number);

			// Choose the position where you want to put the text
			cv::Point textPosition(clusterCenters[i].x, clusterCenters[i].y);

			// Choose font type, scale, and color
			int fontType = cv::FONT_HERSHEY_SIMPLEX;
			float fontScale = 0.4;
			int thickness = 2;
			cv::Scalar color(0, 255, 0); // Green color

			cv::putText(display, text, textPosition, fontType, fontScale, color, thickness);


		}
		cv::imshow("Slic Image: Lowest Grad ", display);

		if (iterationsCounter >= 20) {
			stop = false;
		}

	} while (maxDistanceMoved > minDistanceThreshold || iterationsCounter < maxIterations);

	for (int i = 0; i < clusterCenters.size(); i++) {
		cv::Point location_xy(clusterCenters[i].x, clusterCenters[i].y);
		//cv::Vec3b pixel = input.at<cv::Vec3b>(location_xy.y, location_xy.x);	
		cv::Vec3b blue = { 255, 0, 0 };
		cv::circle(input_circles2, cv::Point(location_xy.x, location_xy.y), 3, blue, -1);

	}
	cv::imshow("Slic Image: Center Difference ", input_circles2);

	cv::Mat normalizedIndexer;
	cv::normalize(indexer, normalizedIndexer, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	cv::Mat output_colorizer = src.clone();
	for (int rows = 0; rows < indexer.rows; rows++) {
		for (int cols = 0; cols < indexer.cols; cols++) {
			// TODO fill the output_colorizer with average color of clusterCenters based on the Indexer values (which are indexes of ClusterCenters)
			int clusterIndex = static_cast<int>(indexer.at<uchar>(rows, cols));
			ClusterCenter center = clusterCenters[clusterIndex];
			cv::Vec3b avgColor(center.B, center.G, center.R);
			output_colorizer.at<cv::Vec3b>(rows, cols) = avgColor;
		}
	}
	cv::imshow("Slic Image: Lowest Grad ", output_colorizer);
	//cv::imshow("Slic Image: Lowest Grad ", normalizedIndexer);
	cv::waitKey(0);
}

void Slic::getBoundaries(int x, int y, int src_rows, int src_cols, int S, int& start_x, int& end_x, int& start_y, int& end_y)
{
	if ((x - S) >= 0) {
		start_x = x - S;
	}
	else {
		start_x = 0;
	}

	if ((x + S) < src_cols) {
		end_x = x + S;
	}
	else {
		end_x = src_cols;
	}

	if ((y - S) >= 0) {
		start_y = y - S;
	}
	else {
		start_y = 0;
	}

	if ((y + S) < src_rows) {
		end_y = y + S;
	}
	else {
		end_y = src_rows;
	}
}

float Slic::euclideanDistance(ClusterCenter& a, ClusterCenter& b)
{
	float distance = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    return distance;
}

cv::Point Slic::findLowestGradientPosition(cv::Mat& src, cv::Point clusterCenterPosition)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	cv::Mat Gx, Gy;
	cv::Sobel(gray, Gx, CV_32F, 1, 0);
	cv::Sobel(gray, Gy, CV_32F, 0, 1);

	cv::Mat G;
	cv::magnitude(Gx, Gy, G);

	cv::Point min(
		std::max(clusterCenterPosition.x - 1, 0), 
		std::max(clusterCenterPosition.y - 1, 0));
	cv::Point max(
		std::min(clusterCenterPosition.x + 1, src.cols - 1), 
		std::min(clusterCenterPosition.y + 1, src.rows - 1));

	float min_gradient = FLT_MAX;
	cv::Point minGradientPosition(
		clusterCenterPosition.x, 
		clusterCenterPosition.y);

	for (int i = min.y; i <= max.y; ++i) {
		for (int j = min.x; j <= max.x; ++j) {
			float currentGradient = G.at<float>(i, j);
			if (currentGradient < min_gradient) {
				min_gradient = currentGradient;
				minGradientPosition = cv::Point(j, i);
			}
		}
	}

	return minGradientPosition;
}

cv::Point Slic::findLowestGradientPosition(cv::Mat& src, int x, int y)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	cv::Mat Gx, Gy;
	cv::Sobel(gray, Gx, CV_32F, 1, 0);
	cv::Sobel(gray, Gy, CV_32F, 0, 1);

	cv::Mat G;
	cv::magnitude(Gx, Gy, G);

	int min_x = std::max(x - 1, 0);
	int min_y = std::max(y - 1, 0);
	int max_x = std::min(x + 1, src.cols - 1);
	int max_y = std::min(y + 1, src.rows - 1);

	float min_gradient = FLT_MAX;
	cv::Point min_gradient_position(x, y);

	for (int i = min_y; i <= max_y; ++i) {
		for (int j = min_x; j <= max_x; ++j) {
			float current_gradient = G.at<float>(i, j);
			if (current_gradient < min_gradient) {
				min_gradient = current_gradient;
				min_gradient_position = cv::Point(j, i);
			}
		}
	}

	return min_gradient_position;
}

void Slic::getBoundaries(cv::Mat src, cv::Point clusterCenterPosition, cv::Point startPixel, cv::Point endPixel, int S)
{
	if ((clusterCenterPosition.x - S) >= 0) {
		startPixel.x = clusterCenterPosition.x - S;
	}
	else {
		startPixel.x = 0;
	}

	if ((clusterCenterPosition.x + S) < src.cols) {
		endPixel.x = clusterCenterPosition.x + S;
	}
	else {
		endPixel.x = src.cols;
	}

	if ((clusterCenterPosition.y - S) >= 0) {
		startPixel.y = clusterCenterPosition.y - S;
	}
	else {
		startPixel.y = 0;
	}

	if ((clusterCenterPosition.y + S) < src.rows) {
		endPixel.y = clusterCenterPosition.y + S;
	}
	else {
		endPixel.y = src.rows;
	}
}

float Slic::recalculateClusters(cv::Mat& src, cv::Mat& indexer, std::vector<ClusterCenter>& clusterCenters)
{
	std::vector<ClusterCenter> newClusterCenters(clusterCenters.size());
	std::vector<int> clusterCenterCounts(clusterCenters.size(), 0);

	for (int y = 0; y < indexer.rows; y++) {
		for (int x = 0; x < indexer.cols; x++) {
			int clusterIndex = static_cast<int>(indexer.at<uchar>(y, x));

			cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
			newClusterCenters[clusterIndex].R += pixel[2];
			newClusterCenters[clusterIndex].G += pixel[1];
			newClusterCenters[clusterIndex].B += pixel[0];
			newClusterCenters[clusterIndex].x += x;
			newClusterCenters[clusterIndex].y += y;

			clusterCenterCounts[clusterIndex]++;
		}
	}

	for (int i = 0; i < clusterCenters.size(); i++) {
		if (clusterCenterCounts[i] > 0) {
			newClusterCenters[i].R /= clusterCenterCounts[i];
			newClusterCenters[i].G /= clusterCenterCounts[i];
			newClusterCenters[i].B /= clusterCenterCounts[i];
			newClusterCenters[i].x /= clusterCenterCounts[i];
			newClusterCenters[i].y /= clusterCenterCounts[i];
		}
	}



	float maxDistanceMoved = 0.0;
	for (int i = 0; i < clusterCenters.size(); i++) {
		float distance = euclideanDistance(clusterCenters[i], newClusterCenters[i]);
		if (distance > maxDistanceMoved) {
			maxDistanceMoved = distance;
		}
	}
	clusterCenters = newClusterCenters;


	return maxDistanceMoved;
}
