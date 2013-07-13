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

void search(string filename){

    Mat img = imread(filename, 1);
    if (img.empty()) {
	printf("Error: image '%s' is empty!\n", filename.c_str());
	return;
    }


    vector<Mat> possible_patentes = patente.search_patent(img,3);
    if(possible_patentes.size() > 0){
        Mat img_patente = possible_patentes[0];

        vector<int> int_characters = patente.search_final_patent(possible_patentes);
        vector<string> string_characters = patente.get_string_characters_from_int(int_characters);

        string patente_characters = "";
        for(int i = 0; i < string_characters.size(); i++){
            if(i%2 == 0)
                patente_characters += " ";

            patente_characters += string_characters[i];
        }

        cout<<patente_characters<<endl;
    }
    else{
	cout<<"Patente not found"<<endl;
    }

}

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
		else if(param.compare("-np") == 0)
			patente.cut_no_patente(argv[2]);
		else if(param.compare("-s") == 0)
			search(argv[2]);

	}

}
