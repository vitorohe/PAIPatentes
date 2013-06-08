#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
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



Mat calculateHist(const string& imageFilename){
	Mat imageData = imread(imageFilename, 1);
	if (imageData.empty()) {
		printf("Error: image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
		return Mat();
	}
	vector<Mat> bgr_planes;
	split( imageData, bgr_planes );

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
	

	/// Normalize the result to [ 0, histImage.rows ]
	// normalize(b_hist, b_hist, 0, 256*3, NORM_MINMAX, -1, Mat() );
	// normalize(g_hist, g_hist, 0, 256*3, NORM_MINMAX, -1, Mat() );
	// normalize(r_hist, r_hist, 0, 256*3, NORM_MINMAX, -1, Mat() );

	Mat hist(1,256*3,CV_32F);
	
	for (int i = 0; i < 256*3; ++i)
	{
		if(i < 256){
			hist.at<float>(0,i) = b_hist.at<float>(i,0);
		}
		else if(i < 512){
			hist.at<float>(0,i) = g_hist.at<float>(i%256,0);
		}
		else{
			hist.at<float>(0,i) = r_hist.at<float>(i%256,0);
		}
	}
	return hist;

}

void addHistToTrainingData(Mat hist, Mat trainingDataMat, int index){
	for (int i = 0; i < 256*3; ++i)
	{
		trainingDataMat.at<float>(index,i) = hist.at<float>(0,i);
	}
}

void train(string filename_to_test){
	cout<<"Training Data"<<endl;

	string dir, filepath;
	int num;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	float labels[220];
	for (int i = 0; i < 220; ++i)
	{
		if (i < 110){
			labels[i] = 1.0;
		}
		else{
			labels[i] = -1.0;
		}
	}
	
	Mat labelsMat(220, 1, CV_32FC1, labels);
	Mat trainingDataMat(220, 256*3, CV_32FC1);


	int index = 0;

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
		addHistToTrainingData(calculateHist(filepath),trainingDataMat,index++);
	}

	closedir( dp );

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
		addHistToTrainingData(calculateHist(filepath),trainingDataMat,index++);
	}

	closedir( dp );

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.degree = 3;
	params.gamma = 3;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

	// cout<<labelsMat<<endl;
	// cout<<trainingDataMat<<endl;

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	if(filename_to_test == "")
		return;

	cout<<"Testing file"<<endl;

	Mat hist_test = calculateHist(filename_to_test);

	float response = SVM.predict(hist_test);

	if (response == 1)
		cout<<"The image "<<filename_to_test<<" IS patente"<<endl;
	else if (response == -1)
		cout<<"The image "<<filename_to_test<<" is NOT patente"<<endl;

}

void search_patent(){

}


int main(int argc, char *argv[]){

	if(argc == 2){
		string param = argv[1];
		if(param.compare("-T") == 0)
			train("");
	}
	else if(argc == 3){
		string param = argv[1];
		if(param.compare("-t") == 0)
			train(argv[2]);
		else if(param.compare("-s") == 0)
			search_patent(argv[2]);
	}

}

// int main()
// {
//     // Data for visual representation
// 	int width = 512, height = 512;
// 	Mat image = Mat::zeros(height, width, CV_8UC3);

//     // Set up training data
// 	float labels[4] = {1.0, -1.0, -1.0, -1.0};
// 	Mat labelsMat(4, 1, CV_32FC1, labels);

// 	float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
// 	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

//     // Set up SVM's parameters
// 	CvSVMParams params;
// 	params.svm_type    = CvSVM::C_SVC;
// 	params.kernel_type = CvSVM::LINEAR;
// 	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

//     // Train the SVM
//  CvSVM SVM;
// 	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

// 	Vec3b green(0,255,0), blue (255,0,0);
//     // Show the decision regions given by the SVM
// 	for (int i = 0; i < image.rows; ++i)
// 		for (int j = 0; j < image.cols; ++j)
// 		{
// 			Mat sampleMat = (Mat_<float>(1,2) << i,j);
// 			float response = SVM.predict(sampleMat);

// 			if (response == 1)
// 				image.at<Vec3b>(j, i)  = green;
// 			else if (response == -1)
// 				image.at<Vec3b>(j, i)  = blue;
// 		}

//     // Show the training data
// 		int thickness = -1;
// 		int lineType = 8;
// 		circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
// 		circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
// 		circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
// 		circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

//     // Show support vectors
// 		thickness = 2;
// 		lineType  = 8;
// 		int c     = SVM.get_support_vector_count();

// 		for (int i = 0; i < c; ++i)
// 		{
// 			const float* v = SVM.get_support_vector(i);
// 			circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
// 		}

//     imwrite("result.png", image);        // save the image

//     imshow("SVM Simple Example", image); // show it to the user
//     waitKey(0);

// }

// int main(){
// 	ifstream fin;
// 	string dir, filepath;
// 	int num;
// 	DIR *dp;
// 	struct dirent *dirp;
// 	struct stat filestat;

// 	// cout << "dir to get files of: " << flush;
// 	// getline( cin, dir );  // gets everything the user ENTERs
// 	dir = "patentes";

// 	dp = opendir( dir.c_str() );
// 	if (dp == NULL)
// 	{
// 		cout << "Error(" << errno << ") opening " << dir << endl;
// 		return errno;
// 	}

// 	while ((dirp = readdir( dp )))
// 	{
// 		filepath = dir + "/" + dirp->d_name;

// 	// If the file is a directory (or is in some way invalid) we'll skip it 
// 		if (stat( filepath.c_str(), &filestat )) continue;
// 		if (S_ISDIR( filestat.st_mode ))         continue;
// 		cout<<dirp->d_name<<endl;
// 	}

// 	closedir( dp );

// 	return 0;
// }


// int main(int argc, char *argv[])
// {
// 	if(argc < 2 || argv == NULL){
//         cout<<"Falta nombre de imagen Ej: ./main autos/image.jpg"<<endl;
// 		return 0;
// 	}
// 	string file = argv[1];
//     string filename = file;

//     Mat imagen = imread(filename, -1);

//     vector<Mat> channels (3);
//     split(imagen, channels);
	
//     if(imagen.empty()){
//         cout<<"ERROR: La imagen "<<filename<<" no pudo ser abierta"<<endl;
//         exit(EXIT_FAILURE);
//     }

//     //imshow("Imagen Original", imagen);

//     Mat binario;
//     //threshold(channels[0], binario, (double)Funciones::umbralOtsu(imagen), 255, THRESH_BINARY);

//     GaussianBlur(channels[0],channels[0],Size(7,7),1.1,1.1,BORDER_DEFAULT);
//     adaptiveThreshold(channels[0], binario, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 15);
//     // imshow("Imagen binaria", binario);
//     Mat segmented;
//     // pyrMeanShiftFiltering(imagen,segmented,8,40,5);
//     // imshow("Imagen Segmentada", segmented);


//     // vector<Mat> channelsSeg (3);
//     // split(segmented, channelsSeg);

//     // imshow("segmented channel", channelsSeg[0]);

//     Mat element_rect = getStructuringElement(MORPH_CROSS, Size(3,3));
//     Mat imagen_bin_dilated;

//     erode(binario,imagen_bin_dilated,element_rect);
//     imshow("erase",imagen_bin_dilated);
//     // vector<vector<Point> > contours;
//     // vector<Vec4i> hierarchy;

//     // Mat canny;
//     // Canny( binario, canny, 100, 100*2, 3 );
//     // imshow("Canny", canny);
//     // findContours( canny, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

//     // /// Draw contours
//     // Mat drawing = Mat::zeros( canny.size(), CV_8UC3 );
//     // RNG rng(12345);
//     // for( int i = 0; i< contours.size(); i++ )
//     //  {
//     //    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//     //    drawContours( drawing, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );
//     //  }

//     // /// Show in a window
//     // namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
//     // imshow( "Contours", drawing );

//     waitKey(0);

//     return 0;
// }
