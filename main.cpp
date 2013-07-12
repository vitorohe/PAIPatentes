#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <cstdlib>
#include "funciones.h"

#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "patente.h"

using namespace std;
using namespace cv;

Patente patente;
SVM_Model svm_model;
MyKNearest knearest;

int main(int argc, char *argv[]){
	patente = Patente();
	svm_model = SVM_Model();
	knearest = MyKNearest();

	if(argc == 2){
		string param = argv[1];
		if(param.compare("-T") == 0)
			svm_model.train();
	}
	else if(argc == 3){
		string param = argv[1];
		if(param.compare("-t") == 0)
			if(svm_model.is_patente(argv[2],Mat(),1)){
				cout<<"IS patente"<<endl;
			}
			else{
				cout<<"is NOT patente"<<endl;
			}
		else if(param.compare("-g") == 0)
			patente.segment_image(argv[2]);
		else if(param.compare("-np") == 0)
			patente.cut_no_patente(argv[2]);
	}
	else if(argc == 4){
		string param = argv[1];
		if(param.compare("-f") == 0){
			int name;
			stringstream ss (argv[3]);
			ss >> name;
			patente.findCharacters(argv[2],name);
		}
		else if(param.compare("-s") == 0)
			patente.search_final_patent(argv[2],argv[3],3);
		else if(param.compare("-fd") == 0)
			patente.feature_detection(argv[2],argv[3]);
	}

}
