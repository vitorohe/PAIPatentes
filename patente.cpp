#include "patente.h"
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

Patente::Patente(){
	svm_model = SVM_Model();
	knearest = MyKNearest();
}

class comparator{
public:
	bool operator()(vector<Point> c1,vector<Point>c2){

		return boundingRect( c1).x<boundingRect( c2).x;

	}

};

class comparator1{
public:
	bool operator()(vector<Point> c1,vector<Point>c2){

		return boundingRect( c1).size().height > boundingRect( c2).size().height;

	}

};

int Patente::extractComponentes(Mat imageSeg, int index, int name) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat canny;
	Mat gray;
	cvtColor(imageSeg,gray,CV_BGR2GRAY);
	adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 5, 2);
//	imshow("imageSeg",gray);
	
	// Canny( gray, canny, 100, 100*2, 3 );
	// imshow("Canny", canny);
	// waitKey(0);
	// return 0;

	Mat element_rect = getStructuringElement(MORPH_RECT, Size(2,1));
	// Mat imagen_bin_dilated;
	Mat imagen_bin_dilated = gray.clone();

	// erode(canny,imagen_bin_dilated,element_rect);
	// dilate(canny,imagen_bin_dilated,element_rect);
	// dilate(imagen_bin_dilated,imagen_bin_dilated,element_rect);

	findContours( imagen_bin_dilated, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	sort(contours.begin(),contours.end(),comparator1());
	/// Draw contours
	Mat drawing = Mat::zeros( imagen_bin_dilated.size(), CV_8UC3 );
	drawing.setTo(Scalar(255,255,255));
	RNG rng(12345);
	Rect rect;
	// if(contours.size() < 6)
	// 	return 0;
	// for( int i = 0; i< contours.size(); i++ )
	// {

		rect = boundingRect(contours[0]);
		cout<<"limit: "<<imageSeg.size().height/2 + imageSeg.size().height/5<<endl;
		cout<<"rect h "<<rect.height<<endl;
		if(rect.height > imageSeg.size().height/2 + imageSeg.size().height/5)
			return -1;
		
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
		// imshow("img_rect",img_rect);
		// waitKey(0);
		// ostringstream string_i;
		// string_i <<"letras/c/comp_"<<name<<"_"<<index++<<".png";
		// string s(string_i.str());
		// imwrite(s,img_rect);
		Scalar color = Scalar( 0,0,0);
		drawContours( drawing, contours, 0, color, CV_FILLED, 8, hierarchy, 0, Point() );
	// }

	/// Show in a window
	//namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	//imshow( "Contours", drawing );
//    waitKey(0);
    // dilate(drawing,drawing,element_rect);
    // dilate(drawing,drawing,element_rect);
	img_rect = drawing(Rect(pt1.x,pt1.y,rect.width,rect.height));
//	imshow("img_rect",img_rect);
//	waitKey(0);
    return knearest.train(img_rect);
//	return 1;
}


vector<Mat> Patente::search_patent(string filename, float factor){

    Mat image = imread(filename, 1);
    if (image.empty()) {
        printf("Error: image '%s' is empty, features calculation skipped!\n", filename.c_str());
        return vector<Mat>();
    }

	cout<<"Looking for patente, factor: "<<factor<<endl;

    int width = image.size().width;
    int height = image.size().height;
	
	int width_rect = width/factor;
	int height_rect = width_rect/2.6;
	// width_rect = width_rect*8/9;
	cout<<"Sliding window: "<<height_rect<<" x "<<width_rect<<endl;
	
	string p;
	Mat rect;
	Mat rect2;
	int index = 0;
	int w_step = 0;
	int height_init;
	
	// if(type == 1){
		height_init = height/3;
		w_step = width_rect/3;
	// }
	// else if(type == 2){
	// 	height_init = 0;
	// 	w_step = width_rect/12;
	// }
	
	vector<int> height_steps;

	// for (int i = 6; i < 11; i = i+2)
	// {
	// 	height_steps.push_back((height_rect/20)*i);
	// }
	height_steps.push_back(height_rect/3);

	vector<Mat> possible_patentes;

	for (int j = 0; j + width_rect < 4*width/5; j = j+w_step)
	{
		for(int n = height_steps.size()-1; n >= 0 ;n--)
		{
			for (int i = height-1-height_rect; i > height_init; i = i-height_steps[n])
			{
                rect = image(Rect(j,i,width_rect,height_rect));
				cvtColor(rect,rect,CV_BGR2GRAY);
				threshold(rect, rect, (double)Funciones::umbralOtsu(rect), 255, THRESH_BINARY);
				ostringstream string_i;
				string_i << index;
				string s(string_i.str());

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
				// imshow("rect2",rect2);
				// waitKey(0);
				if(!svm_model.is_patente("",rect2,2))
					continue;
				// extractComponentes(rect2, 0, index);
				index++;
				imwrite("partes/parte" + s + ".jpg",rect2);
				// add possible patente to vector
				possible_patentes.push_back(rect2);
			}
            if(possible_patentes.size() > 0){
                cout<<"Patente has been found"<<endl;
				break;
            }
		}
	}

	if (index == 0 && factor < 9){
        return search_patent(filename, factor+0.5);
	}else{
		return possible_patentes;
	}
}

void Patente::search_final_patent(vector<Mat> possible_patentes){
    cout<<"Processing Patentes"<<endl;

	// int k = 0;
	
//	Mat image2 = imread("/home/ivan/Documentos/Ramos_U/ProcesamientoAnalisisImagenes/PAIPatentes/patente0046.jpg", 1);
//	if (image2.empty()) {
//		return;
//	}
//	possible_patentes.push_back(image2);

    Mat rect1;
	for(int i = 0; i < possible_patentes.size(); i++){
		cout<<"Patente "<<i<<endl;
//		imshow("possible pat", possible_patentes[i]);
//		waitKey(0);
		// for(int k = 0; k < factors.size(); k++){
		// 	more_possible_patentes = search_patent(possible_patentes[i],possible_patentes[i],factors[k],2);

		// 	for(int j = 0; j < more_possible_patentes.size(); j++){
		// 		imshow("more possible pat", more_possible_patentes[j]);
		// 		waitKey(0);
		// 	}
		// }
		// for (int j = 0; j < 10; ++j)
		// {
		// 	rect1 = possible_patentes[i](Rect(j,0,possible_patentes[i].size().width-1-j,possible_patentes[i].size().height));
		// 	if(svm_model.is_patente("",rect1,2)){
		// 		imshow("more possible pat", rect1);
		// 		waitKey(0);
		// 	}
		// }
		int init = 0;
		int result;

		for (int j = 0; j < 30; j += 3)
		{
			rect1 = possible_patentes[i](Rect(j,0,possible_patentes[i].size().width/7+2,possible_patentes[i].size().height));
//			imshow("more possible pat", rect1);
//			waitKey(0);
			result = extractComponentes(rect1,0,3);
			if(result != -1){
				init = j;
				break;
			}
		}
		cout<<"init: "<<init<<endl;
		possible_patentes[i] = possible_patentes[i](Rect(init,0,possible_patentes[i].size().width-init-1,possible_patentes[i].size().height-1));
		int fin;

		for (int j = possible_patentes[i].size().width-1; j > possible_patentes[i].size().width/2; j -= 3)
		{
			rect1 = possible_patentes[i](Rect(j-1-possible_patentes[i].size().width/7,0,(possible_patentes[i].size().width/7)+2,possible_patentes[i].size().height));
//			imshow("more possible pat", rect1);
//			waitKey(0);
			result = extractComponentes(rect1,0,3);
			if(result != -1){
				fin = j;
				break;
			}
		}
		possible_patentes[i] = possible_patentes[i](Rect(0,0,fin+1,possible_patentes[i].size().height-1));

//		imshow("possible pat", possible_patentes[i]);
//		waitKey(0);
		// return;
		int k = 1;
		for (int j = init; j + possible_patentes[i].size().width/7+2 < possible_patentes[i].size().width; j += possible_patentes[i].size().width/7+2)
		{
			rect1 = possible_patentes[i](Rect(j,0,possible_patentes[i].size().width/7+2,possible_patentes[i].size().height));
//			imshow("more possible pat", rect1);
//			waitKey(0);
			extractComponentes(rect1,0,3);
			if(j+possible_patentes[i].size().width/7+2 >= possible_patentes[i].size().width)
				j = possible_patentes[i].size().width - possible_patentes[i].size().width/7+2;
			if(k%2 == 0)
				j+=3;
			k++;
		}
	}
	cout<<"Searching patente finished"<<endl;

}



int Patente::cut_no_patente(string dir){
	
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

int Patente::extractCharacters(Mat imageSeg, int index, string letras[], int name) {
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

void Patente::findCharacters(string filename,int name){
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

void Patente::segment_image(string filename){
	Mat imagen = imread(filename, -1);

	if(imagen.empty()){
		cout<<"ERROR: La imagen "<<filename<<" no pudo ser abierta"<<endl;
		exit(EXIT_FAILURE);
	}
	resize(imagen, imagen, Size(130,40), 0, 0, INTER_LANCZOS4);
	//imshow("imagen",imagen);
	//waitKey(0);
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

	//imshow( window_name, grad );

	//waitKey(0);
}

void Patente::feature_detection(string filename, string filename2){
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
	//imshow("Matches", img_matches );

	//waitKey(0);
}
