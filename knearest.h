//knearest.h
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class MyKNearest
{
public:
	MyKNearest();
	Mat calculateHist(const string& imageFilename, Mat image, int type);
	void addHistToTrainingData(Mat hist, Mat trainingDataMat, int index);
	void train(Mat input);
};