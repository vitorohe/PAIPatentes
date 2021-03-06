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
    this->setFixedSize(this->size());
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::animateLoading(QLabel *label){
    QMovie *movie = new QMovie("../../style/ajax-loader.gif");
    label->setMovie(movie);
    movie->start();
//    QCoreApplication::processEvents();
}

void MainWindow::searchPatente(Mat img){
//    animateLoading(ui->patente_img);
    patente = Patente();

    vector<Mat> possible_patentes = patente.search_patent(img,3);
    if(possible_patentes.size() > 0){
        Mat img_patente = possible_patentes[0];
        qimgNew = QImage((uchar *)img_patente.data,img_patente.cols,img_patente.rows,img_patente.step,QImage::Format_RGB888).rgbSwapped();
        ui->patente_img->setPixmap(QPixmap::fromImage(qimgNew));
        ui->patente_img->setScaledContents(true);
        QCoreApplication::processEvents();
        vector<int> int_characters = patente.search_final_patent(possible_patentes);
        vector<string> string_characters = patente.get_string_characters_from_int(int_characters);

        string patente_characters = "";
        for(int i = 0; i < string_characters.size(); i++){
            if(i%2 == 0)
                patente_characters += " ";

            patente_characters += string_characters[i];
        }

        ui->patente_label->setText(QString::fromStdString(patente_characters));
    }
    else{
        ui->patente_img->setText("Patente not found");
    }

}

void MainWindow::openImage()
{
    QFileDialog dialog(this);

    dialog.setNameFilter(tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    dialog.setDirectory(QDir::currentPath());
    dialog.setWindowTitle("Open Image");
    QStringList fileNames;
    if (dialog.exec()){
        fileNames = dialog.selectedFiles();
        dialog.close();
    }else{
        return;
    }
    fileName = fileNames.at(0);

    ui->patente_label->setText("XX XX XX");
    ui->patente_img->setText("Looking for...");
    //fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    charFileName = fileName.toLocal8Bit().data();

    Mat img = imread(charFileName,1);
    qimgNew = QImage((uchar *)img.data,img.cols,img.rows,img.step,QImage::Format_RGB888).rgbSwapped();

    ui->lblImage->setPixmap(QPixmap::fromImage(qimgNew));
    ui->lblImage->setScaledContents(true);
    QCoreApplication::processEvents();
    searchPatente(img);
}

void MainWindow::testImage()
{

    QFileDialog dialog(this);

    dialog.setNameFilter(tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    dialog.setDirectory(QDir::currentPath());
    dialog.setWindowTitle("Open Image");
    QStringList fileNames;
    if (dialog.exec()){
        fileNames = dialog.selectedFiles();
        dialog.close();
    }
    else{
        return;
    }
    fileName = fileNames.at(0);

    //fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    charFileName = fileName.toLocal8Bit().data();
    Mat image = imread(charFileName,1);
    qimgNew = QImage((uchar *)image.data,image.cols,image.rows,image.step,QImage::Format_RGB888).rgbSwapped();

    ui->patente_img_test->setPixmap(QPixmap::fromImage(qimgNew));
    ui->patente_img_test->setScaledContents(true);

    if(svm_model.is_patente("",image,2))
        ui->is_patente_label->setText("IS patente");
    else
        ui->is_patente_label->setText("is NOT patente");
}

void MainWindow::trainModel(){
    ui->btnTrain->setEnabled(false);
    ui->train_label->setText("Training...");
    QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
    svm_model = SVM_Model();
    svm_model.train();
    ui->btnTrain->setEnabled(true);
    ui->train_label->setText("Training finished");
}

void MainWindow::on_btnOpen_clicked()
{
    openImage();
}

void MainWindow::on_btnReset_clicked()
{
    ui->lblImage->clear();
    ui->patente_img->setText("Patente");
    ui->patente_label->setText("XX XX XX");
}

void MainWindow::on_btnTrain_clicked()
{
    trainModel();
}

void MainWindow::on_btnTest_clicked()
{
    testImage();
}
