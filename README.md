# classifiers-monorepo

- [classifiers-monorepo](#classifiers-monorepo)
  - [Outline](#outline)
  - [Dependancies](#dependancies)
  - [Supervised Classification](#supervised-classification)
  - [Overview](#overview)

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

[supervised-root]: fixtures/mermaid/supervised-root.png "Supervised Classifiers Based On Learner Type"
