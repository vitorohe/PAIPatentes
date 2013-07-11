#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <opencv2/opencv.hpp>
#include <QFileDialog>

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

void MainWindow::openImage()
{
    fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    charFileName = fileName.toLocal8Bit().data();
    iplImg = cvLoadImage(charFileName);
    qimgNew = QImage((const unsigned char*)iplImg->imageData,iplImg->width,iplImg->height,QImage::Format_RGB888).rgbSwapped();
    ui->lblImage->setPixmap(QPixmap::fromImage(qimgNew));
    ui->lblImage->setScaledContents(true);
    patente = Patente();
    patente.search_final_patent(charFileName,charFileName,3);
}

void MainWindow::toGrayscaleImg()
{
    ui->lblImage->clear();
    IplImage *imgGray = cvLoadImage(charFileName, CV_LOAD_IMAGE_GRAYSCALE);
    qimgGray = QImage((const unsigned char*)imgGray->imageData,imgGray->width,imgGray->height,QImage::Format_Indexed8);
    qimgGray.setPixel(0,0,qRgb(0,0,0));
    ui->lblImage->setPixmap(QPixmap::fromImage(qimgGray));
}

void MainWindow::on_btnOpen_clicked()
{
    openImage();
}

void MainWindow::on_btnReset_clicked()
{
    ui->lblImage->clear();
}

void MainWindow::on_btnToGray_clicked()
{
    toGrayscaleImg();
}
