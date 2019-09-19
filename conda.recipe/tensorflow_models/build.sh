#!/bin/bash

## copy tensorflow/models to conda environment
cp -r $SRC_DIR/models $CONDA_PREFIX/models

## compile protobuf files
START_DIR=`pwd`
cd $CONDA_PREFIX/models/research
protoc --python_out=. ./object_detection/protos/*.proto
cd $START_DIR

## build copy pycocotools to tensorflow/models/research
START_DIR=`pwd`
cd $SRC_DIR/cocoapi/PythonAPI
make
cp -r $SRC_DIR/cocoapi/PythonAPI/pycocotools $CONDA_PREFIX/models/research/pycocotools
cd $START_DIR