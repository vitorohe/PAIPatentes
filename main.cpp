#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

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

int main(){
	ifstream fin;
	string dir, filepath;
	int num;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	// cout << "dir to get files of: " << flush;
	// getline( cin, dir );  // gets everything the user ENTERs
	dir = "patentes";

	dp = opendir( dir.c_str() );
	if (dp == NULL)
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir( dp )))
	{
		filepath = dir + "/" + dirp->d_name;

	// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat( filepath.c_str(), &filestat )) continue;
		if (S_ISDIR( filestat.st_mode ))         continue;
		cout<<dirp->d_name<<endl;
	}

	closedir( dp );

	return 0;
}


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
