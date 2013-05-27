#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdlib>
#include <string>
#include "funciones.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	if(argc < 2 || argv == NULL){
        cout<<"Falta nombre de imagen Ej: ./main autos/image.jpg"<<endl;
		return 0;
	}
	string file = argv[1];
    string filename = file;

    Mat imagen = imread(filename, -1);

    vector<Mat> channels (3);
    split(imagen, channels);
	
    if(imagen.empty()){
        cout<<"ERROR: La imagen "<<filename<<" no pudo ser abierta"<<endl;
        exit(EXIT_FAILURE);
    }

    //imshow("Imagen Original", imagen);

    Mat binario;
    //threshold(channels[0], binario, (double)Funciones::umbralOtsu(imagen), 255, THRESH_BINARY);

    Mat segmented;
    pyrMeanShiftFiltering(imagen,segmented,8,40,5);

    //imshow("Imagen binaria", binario);
    imshow("Imagen Segmentada", segmented);


    vector<Mat> channelsSeg (3);
    split(segmented, channelsSeg);

    imshow("segmented channel", channelsSeg[0]);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat canny;
    Canny( channelsSeg[0], canny, 100, 100*2, 3 );

    findContours( canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours
    Mat drawing = Mat::zeros( segmented.size(), CV_8UC3 );
    RNG rng(12345);
    for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point() );
     }

    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );

    waitKey(0);

    return 0;
}
