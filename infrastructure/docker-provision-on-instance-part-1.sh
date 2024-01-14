# Install docker if it isn't already present
if ! which docker &> /dev/null; then 
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh # throws an error for no reason
    rm get-docker.sh
fi

# Install nvidia-container-toolkit and nvidia-container-runtime
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime

sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

sudo reboot

