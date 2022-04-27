#include "stdafx.h"
#include "MatrixHelper.h"

std::vector<cv::Point> MatrixHelper::getLookAroundMatrix(bool diagonal)
{
	if (diagonal)
	{
		return {
			cv::Point(1, 1), cv::Point(1, 0), cv::Point(1, -1),
			cv::Point(0, 1), /* you are here */	cv::Point(0, -1),
			cv::Point(-1, 1), cv::Point(-1, 0), cv::Point(-1, 1)
		};
	}
	else
	{
		return {
							 cv::Point(1, 0),
			cv::Point(0, 1), /* you are here */	cv::Point(0, -1),
							 cv::Point(-1, 0)
		};
	}
}
