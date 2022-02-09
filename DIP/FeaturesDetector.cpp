#include "stdafx.h"
#include "FeaturesDetector.h"

uchar FeaturesDetector::DetectFeature(uchar input, int threshold)
{
    if (input > threshold) {
        return 1;
    }

    return 0;
}
