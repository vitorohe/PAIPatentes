#ifndef PATENTE_H
#define PATENTE_H

#include "svm_model.h"
#include "knearest.h"

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class Patente
{
public:
	SVM_Model svm_model;
	MyKNearest knearest;

	Patente();
	int extractComponentes(Mat imageSeg, int index, int name);
	vector<Mat> search_patent(Mat imageSeg, Mat image, float factor);
	void search_final_patent(string filename_seg, string filename, float factor);
	int cut_no_patente(string dir);
	int extractCharacters(Mat imageSeg, int index, string letras[], int name);
	void findCharacters(string filename,int name);
	void segment_image(string filename);
	void feature_detection(string filename, string filename2);
};

#endif // PATENTE_H
