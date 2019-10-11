!/bin/bash

#https://medium.com/@jayden.chua/quick-install-cuda-on-google-cloud-compute-6c85447f86a1

sudo apt -y install build-essential cmake unzip pkg-config
sudo apt -y install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo apt -y install libjpeg-dev libpng-dev libtiff-dev
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt -y install libxvidcore-dev libx264-dev
sudo apt -y install libgtk-3-dev
sudo apt -y install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
sudo apt -y install libhdf5-serial-dev
sudo apt -y install python3-dev python3-tk python-imaging-tk
sudo apt -y install gcc-6 g++-6
sudo apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-396
axel https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
axel https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
axel http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda


tar xzvf cudnn-10.0-linux-x64-v7.6.4.38.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
chmod +x ./Anaconda3-2019.07-Linux-x86_64.sh
rm cudnn-10.0-linux-x64-v7.6.4.38.tgz
rm cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >>~/.bashrc
source ~/.bashrc
sudo ./Anaconda3-2019.07-Linux-x86_64.sh
rm Anaconda3-2019.07-Linux-x86_64.sh
source ~/.bashrc
conda config --set auto_activate_base false
source ~/.bashrc
source ./pip.sh