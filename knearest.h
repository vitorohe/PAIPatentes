//knearest.h
#ifndef KNEAREST_H
#define KNEAREST_H

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class MyKNearest
{
protected:
    string letras [35] = {"A","B","C","D","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2","3","4","5","6","7","8","9","0"};
public:

	MyKNearest();
	Mat calculateHist(const string& imageFilename, Mat image, int type);
	void addHistToTrainingData(Mat hist, Mat trainingDataMat, int index);
    int train(Mat input);
    vector<string> get_string_characters_from_int(vector<int> int_characters);
};

#endif // KNEAREST_H
