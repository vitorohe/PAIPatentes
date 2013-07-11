//knearest.h
#ifndef KNEAREST_H
#define KNEAREST_H

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class MyKNearest
{
public:
	MyKNearest();
	Mat calculateHist(const string& imageFilename, Mat image, int type);
	void addHistToTrainingData(Mat hist, Mat trainingDataMat, int index);
    int train(Mat input);
};

#endif // KNEAREST_H
