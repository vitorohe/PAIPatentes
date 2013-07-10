//svm_model.cpp
#include "svm_model.h"
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

using namespace std;
using namespace cv;

SVM_Model::SVM_Model(){}

Mat SVM_Model::calculateHist(const string& imageFilename, Mat image, int type){
	// cout<<"hola"<<endl;
	Mat imageSeg;
	if(type == 1){
		imageSeg = imread(imageFilename, 1);
		if (imageSeg.empty()) {
			printf("Error: image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
			return Mat();
		}
	}
	else
		imageSeg = image.clone();
	
	// cout<<"hola2"<<endl;
	vector<Mat> bgr_planes;
	split( imageSeg, bgr_planes );

	/// Establish the number of bins
	int histSize = 256;
	// cout<<"hola3"<<endl;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist, bin_hist, img1_hist, img2_hist, img3_hist, img4_hist, img5_hist, img6_hist;

	/// Compute the histograms:
	// calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	// calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	// calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
	
	Mat binario;
	resize(imageSeg, imageSeg, Size(120,40), 0, 0, INTER_CUBIC);
	GaussianBlur( imageSeg, imageSeg, Size(3,3), 0, 0, BORDER_DEFAULT );
	// cvtColor(imageSeg,binario,CV_BGR2GRAY);
	// threshold(binario, binario, (double)Funciones::umbralOtsu(imageSeg), 255, THRESH_BINARY);
	// adaptiveThreshold(binario, binario, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 15);

	// calcHist( &binario, 1, 0, Mat(), bin_hist, 1, &histSize, &histRange, uniform, accumulate );

	// int width = imageSeg.size().width;
	// int height = imageSeg.size().height;

	// Mat image1(binario, Range(0,height-1), Range(0,(width/2)-1));
	// Mat image2(binario, Range(0,height-1), Range(width/3, (width*2/3)-1));
	// Mat image3(binario, Range(0,height-1), Range(0,width-1));
	
	// Mat image4(binario, Range(0,(height/3)-1), Range(0, width-1));
	// Mat image5(binario, Range(height/3,(height*2/3)-1), Range(0, width-1));
	// Mat image6(binario, Range(height*2/3, height-1), Range(0, width-1));

	// calcHist( &image1, 1, 0, Mat(), img1_hist, 1, &histSize, &histRange, uniform, accumulate );
	// calcHist( &image2, 1, 0, Mat(), img2_hist, 1, &histSize, &histRange, uniform, accumulate );
	// calcHist( &image3, 1, 0, Mat(), img3_hist, 1, &histSize, &histRange, uniform, accumulate );
	
	// calcHist( &image4, 1, 0, Mat(), img4_hist, 1, &histSize, &histRange, uniform, accumulate );
	// calcHist( &image5, 1, 0, Mat(), img5_hist, 1, &histSize, &histRange, uniform, accumulate );
	// calcHist( &image6, 1, 0, Mat(), img6_hist, 1, &histSize, &histRange, uniform, accumulate );

	/// Normalize the result to [ 0, histImage.rows ]
	// normalize(b_hist, b_hist, 0, 256, NORM_MINMAX, -1, Mat() );
	// normalize(g_hist, g_hist, 0, 256, NORM_MINMAX, -1, Mat() );
	// normalize(r_hist, r_hist, 0, 256, NORM_MINMAX, -1, Mat() );
	// normalize(img1_hist, img1_hist, 0, 256, NORM_MINMAX, -1, Mat() );
	// normalize(img2_hist, img2_hist, 0, 256, NORM_MINMAX, -1, Mat() );
	// normalize(img3_hist, img3_hist, 0, 256, NORM_MINMAX, -1, Mat() );
	// normalize(img4_hist, img4_hist, 0, 256, NORM_MINMAX, -1, Mat() );
	
	// HOGDescriptor hog(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
	//                   Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
	//                   int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
	//                   double threshold_L2hys=0.2, bool gamma_correction=true,
	//                   int nlevels=DEFAULT_NLEVELS);
	HOGDescriptor hog(Size(120, 40), Size(20, 20), Size(5, 5), Size(10, 10), 9);
	vector<float> ders;
	vector<Point>locs;
	
	hog.compute(imageSeg,ders,Size(10,10),Size(0,0),locs);
	Mat hog_mat(ders.size(), 1, CV_32F);
	for (int i = 0; i < ders.size(); ++i)
	{
		hog_mat.at<float>(i,0) = ders.at(i);
	}
	// cout<<ders.size()<<endl;
	// cout<<hog_mat<<endl;
	// exit(0);

	// Mat hist(1,256*6,CV_32F);
	Mat hist(1,3780,CV_32F);
	
	// for (int i = 0; i < 256*6; ++i)
	for (int i = 0; i < 3780; ++i)
	{
		hist.at<float>(0,i) = hog_mat.at<float>(i,0);
		// if(i < 256){
		// 	// hist.at<float>(0,i) = b_hist.at<float>(i,0);
		// 	hist.at<float>(0,i) = img1_hist.at<float>(i,0);
		// }
		// else if(i < 256*2){
		// 	// hist.at<float>(0,i) = g_hist.at<float>(i%256,0);
		// 	hist.at<float>(0,i) = img2_hist.at<float>(i%256,0);
		// }
		// else if(i < 256*3){
		// 	// hist.at<float>(0,i) = r_hist.at<float>(i%256,0);
		// 	hist.at<float>(0,i) = img3_hist.at<float>(i%256,0);
		// }
		// else if(i < 256*4){
		// 	// hist.at<float>(0,i) = bin_hist.at<float>(i%256,0);
		// 	hist.at<float>(0,i) = img4_hist.at<float>(i%256,0);
		// }		
		// else if(i < 256*5){
		// 	// hist.at<float>(0,i) = img1_hist.at<float>(i%256,0);
		// 	hist.at<float>(0,i) = img5_hist.at<float>(i%256,0);
		// }		
		// else if(i < 256*6){
		// 	// hist.at<float>(0,i) = img2_hist.at<float>(i%256,0);
		// 	hist.at<float>(0,i) = img6_hist.at<float>(i%256,0);
		// }		
		// else if(i < 256*7){
			// hist.at<float>(0,i) = img3_hist.at<float>(i%256,0);
			// hist.at<float>(0,i) = bin_hist.at<float>(i%256,0);
		// }
		// else{
		// 	hist.at<float>(0,i) = img4_hist.at<float>(i%256,0);
		// }
	}
	return hist;

}

void SVM_Model::addHistToTrainingData(Mat hist, Mat trainingDataMat, int index){
	// for (int i = 0; i < 256*6; ++i)
	for (int i = 0; i < 3780; ++i)
	{
		trainingDataMat.at<float>(index,i) = hist.at<float>(0,i);
	}
}

void SVM_Model::histToSVMFile(const char* filename, Mat hist, Mat labelsMat){
	ofstream data;
	data.open (filename);
	for (int i = 0; i < hist.size().height; ++i)
	{
		data << labelsMat.at<float>(i,0) <<".0";
		for(int j=0; j < hist.size().width; ++j){
			data << " " << (j+1) <<":"<<hist.at<float>(i,j);
		}
		data << "\n";
	}
	data.close();
}

void SVM_Model::train(){
	// cout<<"Training Data"<<endl;

	string dir, filepath;
	int num;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	float labels[110+2330];
	for (int i = 0; i < 110+2330; ++i)
	{
		if (i < 110){
			labels[i] = 1.0;
		}
		else{
			labels[i] = 0.0;
		}
	}
	
	Mat labelsMat(110+2330, 1, CV_32FC1, labels);
	// Mat trainingDataMat(220, 256*6, CV_32FC1);
	Mat trainingDataMat(110+2330, 3780, CV_32FC1);


	int index = 0;
	// cout<<"Training patentes"<<endl;
	dir = "patentes";

	dp = opendir( dir.c_str() );
	if (dp == NULL)
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
	}

	while ((dirp = readdir( dp )))
	{
		filepath = dir + "/" + dirp->d_name;

	// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		addHistToTrainingData(calculateHist(filepath,Mat(),1),trainingDataMat,index++);
	}

	closedir( dp );
	// cout<<"Training no patentes"<<endl;
	dir = "no_patente";

	dp = opendir( dir.c_str() );
	if (dp == NULL)
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
	}

	while ((dirp = readdir( dp )))
	{
		filepath = dir + "/" + dirp->d_name;

	// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		addHistToTrainingData(calculateHist(filepath,Mat(),1),trainingDataMat,index++);
	}

	closedir( dp );

	histToSVMFile("patente_no_patente.lsvm",trainingDataMat,labelsMat);
	int op = system("libsvm-small/svm-train -s 0 -t 1 -d 2 -g 2.0 -c 1.0 -e 0.000001 -m 500 -b 1 -q patente_no_patente.lsvm patente_no_patente.model");// > /dev/null");
	// cout<<"Training result "<<op<<endl;
	// return;
	/// Set up SVM's parameters
	// CvSVMParams params;
	// params.svm_type    = CvSVM::C_SVC;
	// params.kernel_type = CvSVM::POLY;
	// params.degree = 2;
	// params.p = 0;
	// params.nu = 0;
	// params.C = 1;
	// params.coef0 = 0;
	// params.gamma = 2;
	// params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-6);
	// cout<<labelsMat<<endl;
	// cout<<trainingDataMat<<endl;

	/// Train the SVM
	// CvSVM SVM;
	// SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	// SVM.save("model.xml");
}
bool SVM_Model::is_patente(string filename_to_test, Mat image_to_test, int type){
	// CvSVM SVM;
	// SVM.load("model.xml");
	if(type == 0)
		return false;
	else{
		Mat hist_test;

		if(type == 1){
			// cout<<"Testing file"<<endl;
			hist_test = calculateHist(filename_to_test,Mat(),1);
		}
		else if(type == 2){
			// cout<<"Testing mat"<<endl;
			hist_test = calculateHist("",image_to_test,2);
		}
		Mat labelsMat = Mat::zeros(1,1,CV_32FC1);

		// float result = SVM.predict(hist_test);

		histToSVMFile("test.lsvm",hist_test,labelsMat);
		
		system("libsvm-small/svm-predict -b 1 test.lsvm patente_no_patente.model result > /dev/null");
		
		float class1, class2;

		ifstream infile("result");
		string line;
		int i = 1;
		int j = 0;
		while (getline(infile, line))
		{
			istringstream iss(line);
			int a, b;
			if (i == 2){
				// cout<<iss.str()<<endl;
				string temp;
				while (iss >> temp){
					if(j == 1){
						istringstream ss(temp);
						ss >> class1;
					}
					else if(j == 2){
						istringstream ss(temp);
						ss >> class2;
					}
					j++;
				}
				break;
			}

			i++;
		}

		// remove("test.lsvm");
		// remove("result");
		// cout<<"class1: "<<class1<<", class2: "<<class2<<endl;
		if (class1 > 0.8){
		// if (result == 1){
			// cout<<"The image IS patente"<<endl;
			return true;
		}
		else {
			// cout<<"The image is NOT patente"<<endl;
			return false;
		}
	}

}