#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QMovie>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::animateLoading(){
    QMovie *movie = new QMovie("../../style/ajax-loader.gif");
//    QLabel *processLabel = new QLabel(this);
    ui->patente->setMovie(movie);
    movie->start();
}

void MainWindow::openImage()
{
    fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    charFileName = fileName.toLocal8Bit().data();

    Mat img = imread(charFileName,1);
    qimgNew = QImage((uchar *)img.data,img.cols,img.rows,img.step,QImage::Format_RGB888).rgbSwapped();

    ui->lblImage->setPixmap(QPixmap::fromImage(qimgNew));
    ui->lblImage->setScaledContents(true);

    animateLoading();

    patente = Patente();

    vector<Mat> possible_patentes = patente.search_patent(charFileName,3);
    Mat img_patente = possible_patentes[0];
    qimgNew = QImage((uchar *)img_patente.data,img_patente.cols,img_patente.rows,img_patente.step,QImage::Format_RGB888).rgbSwapped();
    ui->patente->setPixmap(QPixmap::fromImage(qimgNew));
    ui->patente->setScaledContents(true);
    patente.search_final_patent(possible_patentes);
}

void MainWindow::testImage()
{
    fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    charFileName = fileName.toLocal8Bit().data();
    iplImg_test = cvLoadImage(charFileName);
    Mat image(iplImg_test);
    svm_model.is_patente("",image,2);
}

void MainWindow::trainModel(){
    ui->btnTrain->setEnabled(false);
    svm_model = SVM_Model();
    svm_model.train();
    ui->btnTrain->setEnabled(true);
}

void MainWindow::on_btnOpen_clicked()
{
    openImage();
}

void MainWindow::on_btnReset_clicked()
{
    ui->lblImage->clear();
}

void MainWindow::on_btnTrain_clicked()
{
    trainModel();
}

void MainWindow::on_btnTest_clicked()
{
    testImage();
}
