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
    int extractComponentes(Mat imageSeg);
    vector<Mat> search_patent(Mat img, float factor);
    vector<int> search_final_patent(vector<Mat> possible_patentes);
	int cut_no_patente(string dir);
	int extractCharacters(Mat imageSeg, int index, string letras[], int name);
	void findCharacters(string filename,int name);
    vector<string> get_string_characters_from_int(vector<int> int_characters);
};

#endif // PATENTE_H
