#!/bin/bash

conda init bash
conda create -y --name deep-learning python=3.7 && conda activate deep-learning
pip install numpy
pip install opencv-contrib-python
pip install scipy matplotlib pillow
pip install imutils h5py requests progressbar2
pip install scikit-learn scikit-image
pip install tensorflow-gpu==2.0.0
pip install keras
