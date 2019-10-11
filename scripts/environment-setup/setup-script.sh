#!/bin/bash

sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
curl -fsSL https://syncthing.net/release-key.txt | sudo apt-key add -
curl -fsSL https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb [arch=amd64] https://download.docker.com/linux/debian stretch stable" | sudo tee /etc/apt/sources.list.d/docker.list
echo "deb https://apt.syncthing.net/ syncthing candidate" | sudo tee /etc/apt/sources.list.d/syncthing.list
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
source /etc/os-release
curl -fsSL https://deb.nodesource.com/setup_12.x | sudo bash -
sudo apt install -y \
    gcc \
    g++ \
    make \
    gconf2 \
    bsdtar \
    ufw \
    unzip \
    syncthing \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    nodejs \
    yarn \
    nano 

sudo apt install --fix-missing
sudo apt -y --fix-broken install
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo wget https://releases.hashicorp.com/consul/1.6.1/consul_1.6.1_linux_amd64.zip
sudo wget https://releases.hashicorp.com/vault/1.2.3/vault_1.2.3_linux_amd64.zip
sudo wget https://releases.hashicorp.com/terraform/0.12.10/terraform_0.12.10_linux_amd64.zip
sudo wget https://releases.hashicorp.com/packer/1.4.4/packer_1.4.4_linux_amd64.zip

sudo wget https://github.com/containous/traefik/releases/download/v1.7.14/traefik_linux-amd64
sudo unzip ./consul_1.6.1_linux_amd64.zip
sudo unzip ./vault_1.2.3_linux_amd64.zip
sudo unzip ./terraform_0.12.10_linux_amd64.zip
sudo unzip ./packer_1.4.4_linux_amd64.zip
chmod +x ./consul
chmod +x ./vault
chmod +x ./terraform
chmod +x ./packer
chmod +x ./traefik_linux-amd64
sudo mv ./consul /usr/local/bin
sudo mv ./vault /usr/local/bin
sudo mv ./terraform /usr/local/bin
sudo mv ./packer /usr/local/bin
sudo mv ./traefik_linux-amd64 /usr/local/bin/traefik
sudo rm ./consul_1.6.1_linux_amd64.zip
sudo rm ./vault_1.2.3_linux_amd64.zip
sudo rm ./terraform_0.12.10_linux_amd64.zip
sudo rm ./packer_1.4.4_linux_amd64.zip
sudo ufw allow 22 && sudo ufw allow 8500 && sudo ufw allow 8200 && sudo ufw allow 8080 && sudo ufw allow 80 && sudo ufw allow syncthing && sudo ufw enable && sudo ufw reload
sudo groupadd python
sudo usermod -aG python $USER
sudo usermod -aG docker $USER
newgrp docker
newgrp python
sudo chown "$USER":"$USER" /home/"$USER"/.local -R
sudo chmod g+rwx "$HOME/.local" -R
# sudo chown "$USER":"$USER" /home/"$USER"/.docker -R
# sudo chmod g+rwx "$HOME/.docker" -R
# sudo systemctl enable docker
echo "export PATH=$PATH:~/.local/bin" >> ~/.bashrc
echo "export GOPATH=$HOME/go" >> ~/.bashrc
echo "export GOROOT=/usr/local/go" >> ~/.bashrc
echo "export PATH=$PATH:$GOROOT/bin:$GOPATH/bin:/snap/bin:~/.yarn/bin" >> ~/.bashrc
echo "export GO111MODULE=off" >> ~/.bashrc
source ~/.bashrc
