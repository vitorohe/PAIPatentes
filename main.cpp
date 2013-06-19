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



Mat calculateHist(const string& imageFilename, Mat image, int type){
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
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
	
	Mat binario;
	resize(imageSeg, imageSeg, Size(120,40), 0, 0, INTER_CUBIC);
	GaussianBlur( imageSeg, imageSeg, Size(3,3), 0, 0, BORDER_DEFAULT );
	cvtColor(imageSeg,binario,CV_BGR2GRAY);
	threshold(binario, binario, (double)Funciones::umbralOtsu(imageSeg), 255, THRESH_BINARY);
	// adaptiveThreshold(binario, binario, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 15);

	calcHist( &binario, 1, 0, Mat(), bin_hist, 1, &histSize, &histRange, uniform, accumulate );

	int width = imageSeg.size().width;
	int height = imageSeg.size().height;

	Mat image1(binario, Range(0,height-1), Range(0,(width/2)-1));
	Mat image2(binario, Range(0,height-1), Range(width/3, (width*2/3)-1));
	Mat image3(binario, Range(0,height-1), Range(0,width-1));
	
	Mat image4(binario, Range(0,(height/3)-1), Range(0, width-1));
	Mat image5(binario, Range(height/3,(height*2/3)-1), Range(0, width-1));
	Mat image6(binario, Range(height*2/3, height-1), Range(0, width-1));

	calcHist( &image1, 1, 0, Mat(), img1_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &image2, 1, 0, Mat(), img2_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &image3, 1, 0, Mat(), img3_hist, 1, &histSize, &histRange, uniform, accumulate );
	
	calcHist( &image4, 1, 0, Mat(), img4_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &image5, 1, 0, Mat(), img5_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &image6, 1, 0, Mat(), img6_hist, 1, &histSize, &histRange, uniform, accumulate );

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

void addHistToTrainingData(Mat hist, Mat trainingDataMat, int index){
	// for (int i = 0; i < 256*6; ++i)
	for (int i = 0; i < 3780; ++i)
	{
		trainingDataMat.at<float>(index,i) = hist.at<float>(0,i);
	}
}

void train(){
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

	ofstream data;
	data.open ("patente_no_patente.lsvm");
	for (int i = 0; i < trainingDataMat.size().height; ++i)
	{
		data << labelsMat.at<float>(i,0) <<".0";
		for(int j=0; j < trainingDataMat.size().width; ++j){
			data << " " << (j+1) <<":"<<trainingDataMat.at<float>(i,j);
		}
		data << "\n";
	}
	data.close();

	// Set up SVM's parameters
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

	// Train the SVM
	// CvSVM SVM;
	// SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	// SVM.save("model.xml");
}
bool is_patente(string filename_to_test, Mat image_to_test, int type){
	CvSVM SVM;
	SVM.load("model.xml");
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

		float response = SVM.predict(hist_test);

		if (response == 1){
			// cout<<"The image IS patente"<<endl;
			return true;
		}
		else if (response == -1){
			// cout<<"The image is NOT patente"<<endl;
			return false;
		}
	}

}

int extractComponentes(Mat imageSeg, int index, int name) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat canny;
	Canny( imageSeg, canny, 100, 100*2, 3 );
	imshow("Canny", canny);

	Mat element_rect = getStructuringElement(MORPH_RECT, Size(2,2));
	Mat imagen_bin_dilated;

	dilate(canny,imagen_bin_dilated,element_rect);
	dilate(imagen_bin_dilated,imagen_bin_dilated,element_rect);

	findContours( imagen_bin_dilated, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	// sort(contours.begin(),contours.end(),comparator());
	/// Draw contours
	Mat drawing = Mat::zeros( imagen_bin_dilated.size(), CV_8UC3 );
	RNG rng(12345);
	Rect rect;
	if(contours.size() < 6)
		return 0;
	for( int i = 0; i< contours.size(); i++ )
	{

		rect = boundingRect(contours[i]);
		
		Point pt1,pt2;
		pt1.x = rect.x;
		pt1.y = rect.y;
		pt2.x = rect.x + rect.width;
		pt2.y = rect.y + rect.height;

		Mat img_rect = imageSeg(Rect(pt1.x,pt1.y,rect.width,rect.height));
		
		// if(img_rect.size().height > img_rect.size().width/2)
		// 	continue;
		// if(!is_patente("",img_rect,2))
		// 	continue;

		ostringstream string_i;
		string_i <<"letras/c/comp_"<<name<<"_"<<index++<<".png";
		string s(string_i.str());
		imwrite(s,img_rect);
		// Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		// drawContours( drawing, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );
	}

	/// Show in a window
	// namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	// imshow( "Contours", drawing );

    // waitKey(0);
    return index;
}


int search_patent(string filename, string filename_seg, int factor){
	cout<<"Looking for patente, factor: "<<factor<<endl;
	Mat imageSeg = imread(filename_seg, 1);
	if (imageSeg.empty()) {
		printf("Error: image '%s' is empty, features calculation skipped!\n", filename_seg.c_str());
		return 0;
	}

	Mat image = imread(filename, 1);
	if (image.empty()) {
		printf("Error: image '%s' is empty, features calculation skipped!\n", filename.c_str());
		return 0;
	}

	int width = imageSeg.size().width;
	int height = imageSeg.size().height;
	
	int width_rect = width/factor;
	int height_rect = width_rect/3;
	width_rect = width_rect*8/9;

	string p;
	Mat rect;
	Mat rect2;
	int index = 0;

	for (int i = height/3; i + height_rect < height; i = i+height_rect/5)
	{
		for (int j = 0; j + width_rect < width; j = j+height_rect/7)
		{
			rect = imageSeg(Rect(j,i,width_rect,height_rect));
			cvtColor(rect,rect,CV_BGR2GRAY);
			threshold(rect, rect, (double)Funciones::umbralOtsu(rect), 255, THRESH_BINARY);
			ostringstream string_i;
			string_i << index;
			string s(string_i.str());
			bool black = false;
			int b = 0;
			for (int k = 0; k < rect.size().height; ++k)
			{
				for (int l = 0; l < rect.size().width; ++l)
				{
					if((int)rect.at<uchar>(k,l) == 0){
						b++;
					}
				}
			}

			if (b > 0.7*rect.size().height*rect.size().width){
				continue;
			}

			rect2 = image(Rect(j,i,width_rect,height_rect));
			// Mat pat;
			// cvtColor(rect2,pat,CV_BGR2GRAY);
			// threshold(pat, pat, (double)Funciones::umbralOtsu(pat), 255, THRESH_BINARY);
			if(!is_patente("",rect2,2))
				continue;
			// extractComponentes(rect2, 0, index);
			index++;
			imwrite("partes/parte" + s + ".jpg",rect2);
			
		}
	}

	if (index == 0 && factor < 9){
		search_patent(filename, filename_seg, factor+1);
	}

	return index;
}

int cut_no_patente(string dir){
	
	string filepath;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	int index = 111;

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
		
		Mat image = imread(filepath, 1);
		if (image.empty()) {
			printf("Error: image '%s' is empty, features calculation skipped!\n", filepath.c_str());
			return 0;
		}

		int width = image.size().width;
		int height = image.size().height;
		
		int width_rect = 120;
		int height_rect = 40;

		string p;
		Mat rect;

		for (int i = height/3; i + height_rect < height; i = i+height_rect*2)
		{
			for (int j = 0; j + width_rect < width; j = j+width_rect*2)
			{
				rect = image(Rect(j,i,width_rect,height_rect));
				
				ostringstream string_i;
				string_i << index;
				string s(string_i.str());
				
				if(index < 1000)
					s = "0" + s;
				
				// cout<<s<<endl;
				index++;
				imwrite("no_patente/np" + s + ".jpg",rect);
				
			}
		}
	}
}

class comparator{
public:
	bool operator()(vector<Point> c1,vector<Point>c2){

		return boundingRect( c1).x<boundingRect( c2).x;

	}

};


int extractCharacters(Mat imageSeg, int index, string letras[], int name) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat canny;
	Canny( imageSeg, canny, 100, 100*2, 3 );
	imshow("Canny", canny);

	Mat element_rect = getStructuringElement(MORPH_RECT, Size(2,2));
	Mat imagen_bin_dilated;

	dilate(canny,imagen_bin_dilated,element_rect);
	dilate(imagen_bin_dilated,imagen_bin_dilated,element_rect);

	findContours( imagen_bin_dilated, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	sort(contours.begin(),contours.end(),comparator());
	/// Draw contours
	Mat drawing = Mat::zeros( imagen_bin_dilated.size(), CV_8UC3 );
	RNG rng(12345);
	Rect rect;
	for( int i = 0; i< contours.size(); i++ )
	{

		rect = boundingRect(contours[i]);
		
		Point pt1,pt2;
		pt1.x = rect.x;
		pt1.y = rect.y;
		pt2.x = rect.x + rect.width;
		pt2.y = rect.y + rect.height;

		Mat img_rect = imageSeg(Rect(pt1.x,pt1.y,rect.width,rect.height));

		ostringstream string_i;
		string_i <<"letras/c/"<<letras[index++]<<"_"<<name<<".png";
		string s(string_i.str());
		imwrite(s,img_rect);
		// Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		// drawContours( drawing, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );
	}

	/// Show in a window
	// namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	// imshow( "Contours", drawing );

    // waitKey(0);
    return index;
}

void findCharacters(string filename,int name){
	Mat imageSeg = imread(filename, 1);
	if (imageSeg.empty()) {
		printf("Error: image '%s' is empty, features calculation skipped!\n", filename.c_str());
		return;
	}

	cvtColor(imageSeg,imageSeg,CV_BGR2GRAY);

	int height = imageSeg.size().height;
	int width = imageSeg.size().width;

	Mat image1(imageSeg, Range(0,(height/3)-1), Range(0,width-1));
	Mat image2(imageSeg, Range((height/3),2*(height/3) -1), Range(0,width-1));
	Mat image3(imageSeg, Range(2*(height/3), height-1), Range(0,width-1));

	string letras [36] = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2","3","4","5","6","7","8","9","0"};

	int i = extractCharacters(image1, 0, letras, name);
	i = extractCharacters(image2, i, letras, name);
	i = extractCharacters(image3, i, letras, name);
}

void segment_image(string filename){
	Mat imagen = imread(filename, -1);

	if(imagen.empty()){
		cout<<"ERROR: La imagen "<<filename<<" no pudo ser abierta"<<endl;
		exit(EXIT_FAILURE);
	}
	resize(imagen, imagen, Size(130,40), 0, 0, INTER_LANCZOS4);
	imshow("imagen",imagen);
	waitKey(0);
	return;
	
	string window_name = "Sobel Demo - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	GaussianBlur( imagen, imagen, Size(3,3), 0, 0, BORDER_DEFAULT );
	Mat src_gray;
	/// Convert it to gray
	cvtColor( imagen, src_gray, CV_RGB2GRAY );

	/// Create window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	Mat grad;
	/// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	imshow( window_name, grad );

	waitKey(0);
}

void feature_detection(string filename, string filename2){
	Mat img_1 = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_2 = imread( filename2, CV_LOAD_IMAGE_GRAYSCALE );

	if( !img_1.data || !img_2.data )
	{ return ; }

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect( img_1, keypoints_1 );
	detector.detect( img_2, keypoints_2 );

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute( img_1, keypoints_1, descriptors_1 );
	extractor.compute( img_2, keypoints_2, descriptors_2 );

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	//-- Draw matches
	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

	//-- Show detected matches
	imshow("Matches", img_matches );

	waitKey(0);
}

int main(int argc, char *argv[]){

	if(argc == 2){
		string param = argv[1];
		if(param.compare("-T") == 0)
			train();
	}
	else if(argc == 3){
		string param = argv[1];
		if(param.compare("-t") == 0)
			if(is_patente(argv[2],Mat(),1)){
				cout<<"IS patente"<<endl;
			}
			else{
				cout<<"is NOT patente"<<endl;
			}
		else if(param.compare("-g") == 0)
			segment_image(argv[2]);
		else if(param.compare("-np") == 0)
			cut_no_patente(argv[2]);
	}
	else if(argc == 4){
		string param = argv[1];
		if(param.compare("-f") == 0){
			int name;
			stringstream ss (argv[3]);
			ss >> name;
			findCharacters(argv[2],name);
		}
		else if(param.compare("-s") == 0)
			search_patent(argv[2],argv[3],3);
		else if(param.compare("-fd") == 0)
			feature_detection(argv[2],argv[3]);
	}

}
