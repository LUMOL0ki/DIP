#pragma once
class ColorHelper
{
public:
	static cv::Vec3f generateColor();
	static void generateColors(int count, std::vector<cv::Vec3f>& colors, bool includeBackground = true);
	static cv::Vec3f black();
	static cv::Vec3f white();
	static cv::Vec3f red();
	static cv::Vec3f green();
	static cv::Vec3f blue();
};

