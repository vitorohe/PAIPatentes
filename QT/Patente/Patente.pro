#-------------------------------------------------
#
# Project created by QtCreator 2013-07-11T12:32:06
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Patente
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp\
        ../../funciones.cpp\
        ../../svm_model.cpp\
        ../../knearest.cpp\
        ../../patente.cpp\

HEADERS  += mainwindow.h\
        ../../funciones.h\
        ../../svm_model.h\
        ../../knearest.h\
        ../../patente.h\

FORMS    += mainwindow.ui

CONFIG += link_pkgconfig\
            console
PKGCONFIG += opencv
