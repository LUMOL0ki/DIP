#include "stdafx.h"
#include "slic.h"
#include "colorHelper.h"

Slic::Slic(cv::Mat src, int numberOfSegments, int iterations, float threshold)
{
    this->src = src;
    this->numberOfSegments = numberOfSegments;
	this->iterations = iterations;
	this->threshold = threshold;
    this->clusterCenters = std::vector<ClusterCenter>();
}

void Slic::computeSlic()
{
	cv::Mat srcRegularIntervals = src.clone();
	cv::Mat srcLowestGradients = src.clone();
	cv::Mat indexer = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);

	int pixelSize = src.rows * src.cols;
	int iterationsCounter = 0;
	float maxDistanceMoved;
	float step = initializeSteps(pixelSize, numberOfSegments);

	S = static_cast<int>(std::round(step));
	initializeClusterCenters(src, srcRegularIntervals);
	cv::imshow("regular", srcRegularIntervals);

	moveClusterCentersToLowestGradient(src, srcLowestGradients);

	do 
	{
		for (int i = 0; i < clusterCenters.size(); i++) 
		{
			cv::Point start(0, 0);
			cv::Point end(0, 0);
			getBoundaries(src, clusterCenters[i], start, end, S);

			for (int y = start.y; y < end.y; y++) 
			{
				for (int x = start.x; x < end.x; x++) 
				{
					cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

					float minDistance = DBL_MAX;
					int closestCluster = -1;

					for (int c = 0; c < clusterCenters.size(); c++) 
					{
						int m = 19; // You can adjust this value depending on your application

						float dRGB = sqrt(pow(clusterCenters[c].color.x - pixel[2], 2) + pow(clusterCenters[c].color.y - pixel[1], 2) + pow(clusterCenters[c].color.z - pixel[0], 2));
						float dxy = sqrt((clusterCenters[c].position.x - x) * (clusterCenters[c].position.x - x) + (clusterCenters[c].position.y - y) * (clusterCenters[c].position.y - y));

						float Ds = dRGB + (((float)m / S) * dxy);
						float distance = sqrt(pow(clusterCenters[i].position.x - clusterCenters[c].position.x, 2) + pow(clusterCenters[i].position.y - clusterCenters[c].position.y, 2));

						if (Ds < minDistance) 
						{
							if (distance <= sqrt(S * S + S * S))
							{
								minDistance = Ds;
								closestCluster = c;
							}
						}
					}
					indexer.at<uchar>(y, x) = static_cast<uchar>(closestCluster);
				}
			}
		}
		iterationsCounter++;
		maxDistanceMoved = recalculateClusterCenters(src, indexer, clusterCenters);

		printf("Iteration: %d\n", iterationsCounter);

		cv::Mat display = src.clone();
		for (int i = 0; i < clusterCenters.size(); i++) 
		{
			cv::Point location_xy(clusterCenters[i].position.x, clusterCenters[i].position.y);
			cv::circle(display, cv::Point(location_xy.x, location_xy.y), 3, ColorHelper::red(), -1);
		}

		cv::imshow("after convergence", display);

	} while (maxDistanceMoved > threshold || iterationsCounter < iterations);

	cv::Mat dst = src.clone();

	for (int rows = 0; rows < indexer.rows; rows++) 
	{
		for (int cols = 0; cols < indexer.cols; cols++) 
		{
			int clusterIndex = static_cast<int>(indexer.at<uchar>(rows, cols));
			ClusterCenter center = clusterCenters[clusterIndex];
			cv::Vec3b avgColor(center.color.z, center.color.y, center.color.x);
			dst.at<cv::Vec3b>(rows, cols) = avgColor;
		}
	}
	cv::imshow("Final", dst);
	cv::waitKey(0);
}

void Slic::initializeClusterCenters(cv::Mat src, cv::Mat& regularIntervals)
{
	for (int rows = static_cast<int>(std::round(S / 2)); rows < src.rows; rows += S) 
	{
		for (int cols = static_cast<int>(std::round(S / 2) - 10); cols < src.cols; cols += S)
		{
			cv::circle(regularIntervals, cv::Point(cols, rows), 3, ColorHelper::red(), -1);
			cv::Vec3b pixel = src.at<cv::Vec3b>(rows, cols);
			ClusterCenter centers;
			centers.color = cv::Point3i(pixel[2], pixel[1], pixel[0]);
			centers.position = cv::Point(cols, rows);
			clusterCenters.push_back(centers);

			//printf("\n %d - centers.R: %d, centers.G: %d, centers.B: %d, centers.x: %d, centers.y. %d ", centers.color.x, centers.color.y, centers.color.z, centers.position.x, centers.position.y);
		}
	}
}

float Slic::initializeSteps(int pixelSize, int numberOfSegments)
{
	if (numberOfSegments != 0)
	{
		return sqrt(pixelSize / numberOfSegments);
	}

	return 0.0f;
}

void Slic::moveClusterCentersToLowestGradient(cv::Mat src, cv::Mat&lowestGradients)
{
	for (int i = 0; i < clusterCenters.size(); i++) 
	{
		cv::Point lowestGrad = findLowestGradientPosition(src, clusterCenters[i]);
		cv::Vec3b pixel = src.at<cv::Vec3b>(lowestGrad.y, lowestGrad.x);
		cv::circle(lowestGradients, cv::Point(lowestGrad.x, lowestGrad.y), 3, ColorHelper::red(), -1);
		ClusterCenter centers;
		centers.color = cv::Point3i(pixel[2], pixel[1], pixel[0]);
		centers.position = cv::Point(lowestGrad.x, lowestGrad.y);

		//printf("\n %d - centers.R: %d, centers.G: %d, centers.B: %d, centers.x: %d, centers.y. %d ", i, centers.R, centers.G, centers.B, centers.x, centers.y);
	}
}

float Slic::euclideanDistance(ClusterCenter& a, ClusterCenter& b)
{
	float distance = sqrt(pow(a.position.x - b.position.x, 2) + pow(a.position.y - b.position.y, 2));
    return distance;
}

cv::Point Slic::findLowestGradientPosition(cv::Mat& src, ClusterCenter clusterCenter)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::Mat Gx, Gy;
	cv::Sobel(gray, Gx, CV_32F, 1, 0);
	cv::Sobel(gray, Gy, CV_32F, 0, 1);
	cv::Mat G;
	cv::magnitude(Gx, Gy, G);
	cv::Point min(
		std::max(clusterCenter.position.x - 1, 0), 
		std::max(clusterCenter.position.y - 1, 0));
	cv::Point max(
		std::min(clusterCenter.position.x + 1, src.cols - 1), 
		std::min(clusterCenter.position.y + 1, src.rows - 1));

	float min_gradient = FLT_MAX;
	cv::Point minGradientPosition(
		clusterCenter.position.x, 
		clusterCenter.position.y);

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

void Slic::getBoundaries(cv::Mat src, ClusterCenter clusterCenter, cv::Point& start, cv::Point& end, int S)
{
	if ((clusterCenter.position.x - S) >= 0) 
	{
		start.x = clusterCenter.position.x - S;
	}
	else 
	{
		start.x = 0;
	}

	if ((clusterCenter.position.x + S) < src.cols) 
	{
		end.x = clusterCenter.position.x + S;
	}
	else 
	{
		end.x = src.cols;
	}

	if ((clusterCenter.position.y - S) >= 0) 
	{
		start.y = clusterCenter.position.y - S;
	}
	else 
	{
		start.y = 0;
	}

	if ((clusterCenter.position.y + S) < src.rows) 
	{
		end.y = clusterCenter.position.y + S;
	}
	else 
	{
		end.y = src.rows;
	}
}

float Slic::recalculateClusterCenters(cv::Mat& src, cv::Mat& indexer, std::vector<ClusterCenter>& clusterCenters)
{
	std::vector<ClusterCenter> newClusterCenters(clusterCenters.size());
	std::vector<int> clusterCenterCounts(clusterCenters.size(), 0);

	for (int y = 0; y < indexer.rows; y++) 
	{
		for (int x = 0; x < indexer.cols; x++) 
		{
			int clusterIndex = static_cast<int>(indexer.at<uchar>(y, x));

			cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
			newClusterCenters[clusterIndex].color += cv::Point3i(pixel[2], pixel[1], pixel[0]);
			newClusterCenters[clusterIndex].position += cv::Point(x, y);

			clusterCenterCounts[clusterIndex]++;
		}
	}

	float maxDistanceMoved = 0.0;
	for (int i = 0; i < clusterCenters.size(); i++) 
	{
		if (clusterCenterCounts[i] > 0)
		{
			newClusterCenters[i].color.x /= clusterCenterCounts[i];
			newClusterCenters[i].color.y /= clusterCenterCounts[i];
			newClusterCenters[i].color.z /= clusterCenterCounts[i];
			newClusterCenters[i].position.x /= clusterCenterCounts[i];
			newClusterCenters[i].position.y /= clusterCenterCounts[i];
		}

		float distance = euclideanDistance(clusterCenters[i], newClusterCenters[i]);

		if (distance > maxDistanceMoved) 
		{
			maxDistanceMoved = distance;
		}
	}

	clusterCenters = newClusterCenters;

	return maxDistanceMoved;
}