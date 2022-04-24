#pragma once
class FeaturesDetector
{
public:
	/// <summary>
	/// Thresholding the Image.
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Destination image.</param>
	/// <param name="threshold">Threshold value.</param>
	void imageThresholding(cv::Mat src, cv::Mat& dst, float threshold);
	/// <summary>
	/// Indexing and coloring destination image.
	/// </summary>
	/// <param name="src"></param>
	/// <param name="dst"></param>
	void objectIndexingColored(cv::Mat src, cv::Mat& dst);
	void objectIndexingNumbered(cv::Mat src, cv::Mat& dst);
private:
	/// <summary>
	/// Indexing objects.
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Indexed destination image.</param>
	/// <returns>Number of objects.</returns>
	int objectIndexing(cv::Mat src, cv::Mat& dst);
	void proccesingIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const cv::Point  lookAroundMatrix[8], int lookAroundMatrixSize, int& id);
	void assignIndexes(cv::Mat& src, cv::Mat& dst, std::vector<cv::Point>& pixels, const cv::Point  lookAroundMatrix[8], int lookAroundMatrixSize, int& id);
	void CheckNeighboringPixels(cv::Mat& src, std::vector<cv::Point>& pixels, cv::Point& currentPixel, const cv::Point lookAroundMatrix[8], int lookAroundMatrixSize);
	/// <summary>
	/// 
	/// </summary>
	/// <param name="src">Source image.</param>
	/// <param name="dst">Destination image.</param>
	/// <param name="colors">array of colors</param>
	void assignRandomColors(cv::Mat src, cv::Mat& dst, int count);
};

