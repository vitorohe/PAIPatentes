#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ui_mainwindow.h"
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
    void searchPatente(Mat img);
    void openImage();
    void trainModel();
    void testImage();
    void animateLoading(QLabel *label);
    void on_btnOpen_clicked();
    void on_btnReset_clicked();
    void on_btnTrain_clicked();
    void on_btnTest_clicked();

private:
    Ui::MainWindow *ui;
    QString fileName;
    IplImage *iplImg;
    IplImage iplImg_patente;
    IplImage *iplImg_test;
    Mat image;
    char* charFileName;
    QImage qimgNew;
    QImage qimgGray;
    Patente patente;
    SVM_Model svm_model;
};

#endif // MAINWINDOW_H
