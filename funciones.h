//funciones.h
#ifndef FUNCIONES_H
#define FUNCIONES_H
#include <opencv2/core/core.hpp>

using namespace cv;

class Funciones
{
public:
	Funciones();
	static int umbralOtsu(Mat input);
	static Mat histograma(Mat input);
	static vector<vector<Point> > get_contours(Mat input, vector<Vec4i>& hierarchy);

};

#endif // FUNCIONES_H
