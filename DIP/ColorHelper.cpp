#include "stdafx.h"
#include "colorHelper.h"
#include <random>

cv::Vec3f ColorHelper::generateColor()
{
    // Stackoverflow goes brrrr...
    std::random_device randomDevice;
    std::mt19937 engine(randomDevice());
    std::uniform_int_distribution<> distr(20, 255);
    return cv::Vec3f(
        float(distr(engine)) / 255,
        float(distr(engine)) / 255,
        float(distr(engine)) / 255);
}

void ColorHelper::generateColors(int count, std::vector<cv::Vec3f>& colors, bool includeBackground)
{
    if (includeBackground) 
    {
        colors.push_back(ColorHelper::black());
    }

    for (int i = 0; i < count; i++)
    {
        colors.push_back(ColorHelper::generateColor());
    }
}

cv::Vec3f ColorHelper::black()
{
    return cv::Vec3f();
}

cv::Vec3f ColorHelper::white()
{
    return cv::Vec3f(255.0, 255.0, 255.0);
}

cv::Vec3f ColorHelper::red()
{
    return cv::Vec3f(0.0, 0.0, 255.0);
}

cv::Vec3f ColorHelper::green()
{
    return cv::Vec3f(0.0, 255.0, 0.0);
}

cv::Vec3f ColorHelper::blue()
{
    return cv::Vec3f(255.0, 0.0, 0.0);
}
