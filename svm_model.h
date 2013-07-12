//svm_model.h
#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class SVM_Model
{
public:
	SVM_Model();
	Mat calculateHist(const string& imageFilename, Mat image, int type);
	void addHistToTrainingData(Mat hist, Mat trainingDataMat, int index);
	void histToSVMFile(const char* filename, Mat hist, Mat labelsMat);
	void train();
	bool is_patente(string filename_to_test, Mat image_to_test, int type);
};

#endif // SVM_MODEL_H
