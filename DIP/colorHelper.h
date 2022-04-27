#pragma once
class ColorHelper
{
public:
	static cv::Vec3f generateColor();
	static void generateColors(int count, std::vector<cv::Vec3f>& colors, bool includeBackground = true);
	static cv::Vec3f Black();
	static cv::Vec3f White();
};

