#include "knearest.h"
#include <stdio.h>
#include "ml.h"
#include "highgui.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <cstdlib>

#include <fstream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace cv;

MyKNearest::MyKNearest(){}

Mat MyKNearest::calculateHist(const string& imageFilename, Mat image, int type){
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
	resize(imageSeg, imageSeg, Size(40,60), 0, 0, INTER_CUBIC);
	//GaussianBlur( imageSeg, imageSeg, Size(3,3), 0, 0, BORDER_DEFAULT );
	// cvtColor(imageSeg,binario,CV_BGR2GRAY);
	// threshold(binario, binario, (double)Funciones::umbralOtsu(imageSeg), 255, THRESH_BINARY);
	// adaptiveThreshold(binario, binario, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 15);

	// HOGDescriptor hog(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
	//                   Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
	//                   int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
	//                   double threshold_L2hys=0.2, bool gamma_correction=true,
	//                   int nlevels=DEFAULT_NLEVELS);
	HOGDescriptor hog(Size(40, 60), Size(20, 20), Size(5, 5), Size(10, 10), 9);
	vector<float> ders;
	vector<Point>locs;
	//cout<<"Ante de compute"<<endl;
	hog.compute(imageSeg,ders,Size(10,10),Size(0,0),locs);
	//cout<<"Despues de compute"<<endl;
	Mat hog_mat(ders.size(), 1, CV_32F);
	for (int i = 0; i < ders.size(); ++i)
	{
		hog_mat.at<float>(i,0) = ders.at(i);
	}
	//cout<<ders.size()<<endl;

	// cout<<hog_mat<<endl;
	// exit(0);

	// Mat hist(1,256*6,CV_32F);
	Mat hist(1,1620,CV_32F);
	
	// for (int i = 0; i < 256*6; ++i)
	for (int i = 0; i < 1620; ++i)
	{
		hist.at<float>(0,i) = hog_mat.at<float>(i,0);
	}
	return hist;

}

void MyKNearest::addHistToTrainingData(Mat hist, Mat trainingDataMat, int index){
	// for (int i = 0; i < 256*6; ++i)
	for (int i = 0; i < 1620; ++i)
	{
		trainingDataMat.at<float>(index,i) = hist.at<float>(0,i);
	}
}

int MyKNearest::train(Mat input){
	cout<<"Training Data"<<endl;

	Mat imageDataMat(1, 1620, CV_32FC1);
	addHistToTrainingData(calculateHist("",input,2),imageDataMat,0);
	/*--------------------------------------------------------------------------*/

	string dir, filepath;
	int num;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	float labels[612];
	for (int i = 0, k = -1; i < 612; i++) {
		if(i%17 == 0)
			k++;
		labels[i] = (float)k;
	}
	
	Mat labelsMat(612, 1, CV_32FC1, labels);
	// Mat trainingDataMat(220, 256*6, CV_32FC1);
	Mat trainingDataMat(612, 1620, CV_32FC1);

	// cout<<labelsMat<<endl;

	int index = 0;
	// cout<<"Training patentes"<<endl;
	dir = "../../letras/caracteres";

	/*dp = opendir( dir.c_str() );
	if (dp == NULL)
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
	}

	while ((dirp = readdir( dp )))
	{
		filepath = dir + "/" + dirp->d_name;
		cout<<"File: "<<filepath<<endl;
	// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		addHistToTrainingData(calculateHist(filepath,Mat(),1),trainingDataMat,index++);
	}

	closedir( dp );*/

	for(int l = 0; l < 35; l++) {
		for(int i = 1; i < 18; i++) {
			std::stringstream stream;
		    stream<<dir<<"/"<<letras[l]<<"_"<<i<<".png";

		    filepath = stream.str();
			//cout<<"Calculate Hist for File: "<<filepath<<endl;
			addHistToTrainingData(calculateHist(filepath,Mat(),1),trainingDataMat,index++);
			//cout<<"File: "<<filepath<<endl;
		}
	}

	int K = 10;
	CvKNearest knn( trainingDataMat, labelsMat, Mat(), false, K );
    Mat nearests( 1, K, CV_32FC1);

	// estimate the response and get the neighbors' labels
    float response = knn.find_nearest(imageDataMat,K,0,0,&nearests,0);
    if(response >= 35) {
		cout<<"This is no a letter o number"<<endl;
        return -1;
	}
	cout<<"Response: "<<response<<" = "<<letras[(int)response]<<endl;
    // compute the number of neighbors representing the majority
    for( int k = 0; k < K; k++ )
    {
        cout<<"Vecino: "<<nearests.at<float>(0,k)<<" "<<letras[(int)nearests.at<float>(0,k)]<<endl;
    }

    return (int)response;
}

vector<string> MyKNearest::get_string_characters_from_int(vector<int> int_characters){
    vector<string> string_characters;
    for(int i = 0; i < int_characters.size(); i++){
        if(int_characters[i] != -1)
            string_characters.push_back(letras[int_characters[i]]);
        else
            string_characters.push_back("?");
    }

    return string_characters;
}
