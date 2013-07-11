#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "../../patente.h"

using namespace cv;

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void openImage();
    void toGrayscaleImg();
    void on_btnOpen_clicked();
    void on_btnReset_clicked();
    void on_btnToGray_clicked();
    
private:
    Ui::MainWindow *ui;
    QString fileName;
    IplImage *iplImg;
    Mat image;
    char* charFileName;
    QImage qimgNew;
    QImage qimgGray;
    Patente patente;
};

#endif // MAINWINDOW_H
