//funciones.cpp
#include "funciones.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <typeinfo>

using namespace std;
using namespace cv;


Funciones::Funciones(){}

Mat Funciones::histograma(Mat input){
	Mat hist = Mat::zeros(256,1,CV_32S);
	int j = 0, i = 0;

	for(i = 0; i < input.size().height; i++){
		for(j = 0; j < input.size().width; j++){
			hist.at<int>(input.at<uchar>(i,j),0)++;
		}
	}

	return hist;
}

int Funciones::umbralOtsu(Mat imagen){
	Mat h = Funciones::histograma(imagen);
	h.convertTo(h,CV_32F);
	int i = 0;

	int suma = sum(h)[0];
	
	h= h/suma;
	
	float acum [256];
	float media [256];

	acum[0] = h.at<float>(0,0);
	media[0] = 0;


	for(i = 1; i <= 255; i++){
		acum[i] = h.at<float>(i,0) + acum[i-1];
		media[i] = i*h.at<float>(i,0) + media[i-1];
	}

	float m = media[255];
	float maxVal = 0;
	uchar umbral = 0;
	float val = 0;
	float P1 = 0, P2 = 0, m1 = 0, m2 = 0;

	for(i = 0; i <= 255; i++){
		P1 = acum[i];
		P2 = 1 - P1;
		m1 = media[i]/P1;
		m2 = (m - media[i])/P2;
		val = P1*pow(m1 - m,2) + P2*pow(m2 - m,2);
		if(val > maxVal){
			maxVal = val;
			umbral = i;
		}
	}

	return umbral;
}

vector<vector<Point> > Funciones::get_contours(Mat input, vector<Vec4i>& hierarchy){
	vector<vector<Point> > contours;

	findContours( input, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	return contours;
}
