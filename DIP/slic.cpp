#include "stdafx.h"
#include "slic.h"

void Slic::initializeData(cv::Mat src)
{ 
    // Initialize clusters and distances. 
    for (int y = 0; y < src.rows; y++) {
        std::vector<int> cluster;
        std::vector<double> distance;
        for (int x = 0; x < src.cols; x++) {
            cluster.push_back(-1);
            distance.push_back(FLT_MAX);
        }
        clusters.push_back(cluster);
        distances.push_back(distance);
    }

    // Initialize centers and counters. 
    for (int y = step; y < src.rows - step / 2; y += step) {
        for (int x = step; x < src.cols - step / 2; x += step) {
            std::vector<double> center;
            cv::Point nc = findLocalMinimum(src, cv::Point(x, y));
            cv::Scalar color(src.at<float>(nc.y, nc.x));

            center.push_back(color.val[0]);
            center.push_back(color.val[1]);
            center.push_back(color.val[2]);
            center.push_back(nc.x);
            center.push_back(nc.y);

            centers.push_back(center);
            centersCount.push_back(0);
        }
    }
}

void Slic::generateSuperpixels(cv::Mat& src, int steps, int nc)
{
    this->step = steps;
    this->nc = nc;
    this->distance = steps;

    initializeData(src);
    /* Run EM for 10 iterations (as prescribed by the algorithm). */
    for (int i = 0; i < 10; i++) 
    {
        /* Reset distance values. */
        for (int j = 0; j < src.cols; j++) 
        {
            for (int k = 0; k < src.rows; k++) 
            {
                distances[j][k] = FLT_MAX;
            }
        }

        for (int j = 0; j < (int)centers.size(); j++) 
        {
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int k = centers[j][3] - steps; k < centers[j][3] + step; k++) 
            {
                for (int l = centers[j][4] - steps; l < centers[j][4] + step; l++) 
                {
                    if (k >= 0 && k < src.cols && l >= 0 && l < src.rows) 
                    {
                        cv::Scalar color(src.at<float>(l, k));
                        double d = calculateDistance(j, cv::Point(k, l), color);

                        /* Update cluster allocation if the cluster minimizes the
                           distance. */
                        if (d < distances[k][l]) {
                            distances[k][l] = d;
                            clusters[k][l] = j;
                        }
                    }
                }
            }
        }

        /* Clear the center values. */
        for (int j = 0; j < (int)centers.size(); j++) 
        {
            centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
            centersCount[j] = 0;
        }

        /* Compute the new cluster centers. */
        for (int j = 0; j < src.cols; j++) 
        {
            for (int k = 0; k < src.rows; k++) 
            {
                int c_id = clusters[j][k];

                if (c_id != -1) 
                {
                    cv::Scalar color(src.at<float>(k, j));

                    centers[c_id][0] += color.val[0];
                    centers[c_id][1] += color.val[1];
                    centers[c_id][2] += color.val[2];
                    centers[c_id][3] += j;
                    centers[c_id][4] += k;

                    centersCount[c_id] += 1;
                }
            }
        }

        /* Normalize the clusters. */
        for (int j = 0; j < (int)centers.size(); j++) {
            centers[j][0] /= centersCount[j];
            centers[j][1] /= centersCount[j];
            centers[j][2] /= centersCount[j];
            centers[j][3] /= centersCount[j];
            centers[j][4] /= centersCount[j];
        }
    }
}

void Slic::createConectivity(cv::Mat& src)
{
    int label = 0, adjlabel = 0;
    const int lims = (src.cols * src.rows) / ((int)centers.size());

    const int dx4[4] = { -1,  0,  1,  0 };
    const int dy4[4] = { 0, -1,  0,  1 };

    /* Initialize the new cluster matrix. */
    std::vector<std::vector<int>> new_clusters;
    for (int i = 0; i < src.cols; i++) 
    {
        std::vector<int> nc;
        for (int j = 0; j < src.rows; j++) 
        {
            nc.push_back(-1);
        }
        new_clusters.push_back(nc);
    }

    for (int i = 0; i < src.cols; i++) 
    {
        for (int j = 0; j < src.rows; j++) 
        {
            if (new_clusters[i][j] == -1) 
            {
                std::vector<cv::Point> elements;
                elements.push_back(cv::Point(i, j));

                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];

                    if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
                        if (new_clusters[x][y] >= 0) {
                            adjlabel = new_clusters[x][y];
                        }
                    }
                }

                int count = 1;
                for (int c = 0; c < count; c++) 
                {
                    for (int k = 0; k < 4; k++) 
                    {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];

                        if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) 
                        {
                            if (new_clusters[x][y] == -1 && clusters[i][j] == clusters[x][y]) 
                            {
                                elements.push_back(cvPoint(x, y));
                                new_clusters[x][y] = label;
                                count += 1;
                            }
                        }
                    }
                }

                /* Use the earlier found adjacent label if a segment size is
                   smaller than a limit. */
                if (count <= lims >> 2) 
                {
                    for (int c = 0; c < count; c++) 
                    {
                        new_clusters[elements[c].x][elements[c].y] = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    }
}

double Slic::calculateDistance(int clusterId, cv::Point pixel, cv::Scalar color)
{
    double dc = sqrt(pow(centers[clusterId][0] - color.val[0], 2) + pow(centers[clusterId][1] - color.val[1], 2) + pow(centers[clusterId][2] - color.val[2], 2));
    double ds = sqrt(pow(centers[clusterId][3] - pixel.x, 2) + pow(centers[clusterId][4] - pixel.y, 2));
    return sqrt(pow(dc / nc, 2) + pow(ds / distance, 2));
}

CvPoint Slic::findLocalMinimum(cv::Mat dst, cv::Point center)
{
    double min_grad = FLT_MAX;
    cv::Point localMinimum = cv::Point(center.x, center.y);

    for (int x = center.x - 1; x < center.x + 2; x++) 
    {
        for (int y = center.y - 1; y < center.y + 2; y++) 
        {
            cv::Scalar c1(dst.at<float>(x + 1, y));
            cv::Scalar c2(dst.at<float>(x, y + 1));
            cv::Scalar c3(dst.at<float>(x, y));
            double i1 = c1.val[0];
            double i2 = c2.val[0];
            double i3 = c3.val[0];

            if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3, 2)) < min_grad) 
            {
                min_grad = fabs(i1 - i3) + fabs(i2 - i3);
                localMinimum.x = x;
                localMinimum.y = y;
            }
        }
    }

    return localMinimum;
}

Slic::Slic(cv::Mat src, int steps, int nc)
{
    generateSuperpixels(src, steps, nc);
    createConectivity(src);
}

double Slic::getNumberOfSteps(int width, int height, double superpixels)
{
    return sqrt((width * height) / superpixels);
}

void Slic::drawCenterGrid(cv::Mat& dst, cv::Scalar color)
{
    for (int i = 0; i < (int)centers.size(); i++) 
    {
        cv::circle(dst, cvPoint(centers[i][3], centers[i][4]), 2, color, 2);
    }
}

void Slic::drawContours(cv::Mat& dst, cv::Scalar color)
{
    const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

    std::vector<cv::Point> outlines;
    std::vector<std::vector<bool>> areTaken;
    for (int i = 0; i < dst.cols; i++) 
    {
        std::vector<bool> isTaken;

        for (int j = 0; j < dst.rows; j++) 
        {
            isTaken.push_back(false);
        }

        areTaken.push_back(isTaken);
    }

    /* Go through all the pixels. */
    for (int x = 0; x < dst.cols; x++) 
    {
        for (int y = 0; y < dst.rows; y++) 
        {
            int nr_p = 0;

            // Look around.
            for (int k = 0; k < 8; k++) {
                int x2 = x + dx8[k];
                int y2 = y + dy8[k];

                if (x2 >= 0 && x2 < dst.cols && y2 >= 0 && y2 < dst.rows) {
                    if (areTaken[x2][y2] == false && clusters[x][y] != clusters[x2][y2]) {
                        nr_p++;
                    }
                }
            }

            if (nr_p >= 2) {
                outlines.push_back(cv::Point(x, y));
                areTaken[x][y] = true;
            }
        }
    }

    for (cv::Point outline : outlines) 
    {
        dst.at<cv::Scalar>(outline.y, outline.x) = color;
    }
}

void Slic::drawColorWithClusterMeans(cv::Mat& dst, cv::Scalar color)
{
    std::vector<cv::Scalar> colors(centers.size());

    for (int x = 0; x < dst.cols; x++)
    {
        for (int y = 0; y < dst.rows; y++)
        {
            int clusterId = clusters[x][y];
            cv::Scalar color(dst.at<float>(y, x));

            for (int valId = 0; valId < 3; valId++) 
            {
                colors[clusterId].val[valId] += color.val[valId]; // Collect color values per cluster.
            }
        }
    }

    for (int colorId = 0; colorId < colors.size(); colorId++) 
    {
        for (int valId = 0; valId < 3; valId++) 
        {
            colors[colorId].val[valId] /= centersCount[colorId]; // Get mean color.
        }
    }

    for (int x = 0; x < dst.cols; x++)
    {
        for (int y = 0; y < dst.rows; y++)
        {
            dst.at<cv::Scalar>(y, x) = colors[clusters[x][y]]; // Assaign colors;
        }
    }
}
