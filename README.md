# classifiers-monorepo

- [classifiers-monorepo](#classifiers-monorepo)
  - [Outline](#outline)
  - [Dependancies](#dependancies)
  - [Supervised Classification](#supervised-classification)
  - [Overview](#overview)
  - [Lazy learner](#lazy-learner)
    - [K-Nearest Neighbor](#k-nearest-neighbor)
    - [Case-based Reasoning](#case-based-reasoning)
  - [Eager Learner](#eager-learner)
    - [Decision Tree](#decision-tree)
    - [Logistic Regression](#logistic-regression)
    - [Naive Bayes](#naive-bayes)
    - [Random Forest](#random-forest)
    - [Support Vector Machine](#support-vector-machine)
    - [Artificial Neural Networks](#artificial-neural-networks)
  - [References](#references)

## Outline

This repo is used to host material on machine learning classification methodologies and algorithms.

## Dependancies

To offer consistent workflow, all files and dependencies are compiled in isolated docker container by default when using `gnu make`.
If you want to build and run make target on your own machine , out of docker containers , edit Makefile and set `DOCKER_ENV` to `false`. Make sure to run `make dep` in case you choose to not use the dockerized build pipeline.

All diagrams and flowcharts are generated with [`Mermaid.JS`](https://mermaidjs.github.io) . Source codes for mermaid files are stored under `mermaid` directory and generated pdf files can be found at `fixtures/mermaid`. You can generate diagrams by running `make mermaid` in terminal.

An script for setting up development environment for machine learning is provided at `scripts/environment-setup`. This script has been tested on has been tested on google compute engine and is designed to set up a debian development environment with `python`, `conda`,`numpy`,`tensorflow(GPU)` ,`keras` to name a few and also tries to setup `CUDA`. A docker file will be provided in the future that uses these script to create a container for running all deep learning tasks. If you are planning on using this script to setup your environment ( or build the docker file ), Due to licensisng issues, you must download `cuDNN` from `nVidia` website before running the script and put it in the same directory as the rest of the files (`scripts/environment-setup`). Follow the guide in this [`Medium Article`](https://medium.com/@jayden.chua/quick-install-cuda-on-google-cloud-compute-6c85447f86a1).

## Supervised Classification

![Supervised Classifiers Based On Learner Type][supervised-root]

## Overview

`Lazy learner` simply store the training data and wait up until testing data appears ; they classify at that point based on the most related data in the stored training data. To put it in simpler terms, A lazy learner does not have a training phase. As an example, lazy learners are used by online recommendation systems.

`Eager learner` on the other hand before receiving data for classification setup a classification model which is based on the data passed to it during the training phase. Eager learning algorithms must be able to commit to a single hypothesis that covers the entire instance space. to put it in simpler terms, An eager learner has a model fitting or training step.

|   Classifier  | Time To Train | Time To Predict |
|:-------------:|:-------------:|:---------------:|
|  Lazy Learner |      Low      |       High      |
| Eager Learner |      High     |       Low       |

## Lazy learner

### K-Nearest Neighbor

`KNN` Is one of the more basic algorithms and is oftenly used to benchmark more complex classifiers like Artificial Neural Networks and Support Vector Machines.

It operates by storing all instances correspond to training data points in n-dimensional space. When an unknown discrete data is received, it analyzes the closest k number of instances saved and returns the most common class as the prediction result .In case of real-valued data it returns the mean of k nearest neighbors.

There is also other variation of KNN such as:

- [Distance-weighted KNN](https://www.geeksforgeeks.org/weighted-k-nn/)
- [dual distance-weighted KNN](https://pdfs.semanticscholar.org/a128/62972be0e7e6e901825723e703117a6d8128.pdf) 
- [Modified K-Nearest Neighbor (MKNN)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.545&rep=rep1&type=pdf)
- [Dynamic K-Nearest-Neighbor](https://ieeexplore.ieee.org/abstract/document/5559858)

I also found an inseresting [paper](https://ieeexplore.ieee.org/abstract/document/4406010) that offers different methodologies to pmproving K-Nearest-Neighbor for Classification.

The following are some real-world sample implementations and/or utilization of KNN:

- `[GO]` [Implementation In Pure GO](https://github.com/mervin0502/knnAlg)
- `[GO]` [Webserver For classifying images with KNN](https://github.com/arnaucube/galdric)
- `[RUST]` [KNN Node.js module written in Rust for high perfromance and parallelism](https://github.com/houtanf/Unsupervised-KNN-JS)
- `[RUST]` [KNN algorithm with GPU calculation](https://github.com/stoand/rust-gpu_kNN)
- `[PYTHON]` [Simple Python Implementation of KNN algorithm ](https://github.com/iiapache/KNN)

### Case-based Reasoning

## Eager Learner

### Decision Tree

### Logistic Regression

### Naive Bayes

### Random Forest

### Support Vector Machine

### Artificial Neural Networks

## References

[supervised-root]: fixtures/mermaid/supervised-root.png "Supervised Classifiers Based On Learner Type"

