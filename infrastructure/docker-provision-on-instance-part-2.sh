set -e

MLAB2_IMAGE=mlabredwood/mlab2:latest

sudo setfacl --modify user:ubuntu:rw /var/run/docker.sock

# Add the mlab2_ssh key to the ssh-agent
chmod 600 ~/.ssh/mlab2_ssh
eval `ssh-agent -s`
ssh-add ~/.ssh/mlab2_ssh
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Log in to docker
docker login --username=mlabredwood --password=dckr_pat_P-ma_kUDCWwDka8W7K-ta8iGbUE

# Stop any existing containers
if [ -n "$(docker ps -a -q)" ]; then
    docker kill $(docker ps -a -q)
    sleep 2
fi

# Start one container if there are < 8 GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $NUM_GPUS -lt 8 ]; then
    docker run --name "group-0" --rm -d  -v $(readlink -f $SSH_AUTH_SOCK):/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent --runtime=nvidia --gpus all $MLAB2_IMAGE
fi

# Start two containers if there are >= 8 GPUs
if [ $NUM_GPUS -ge 8 ]; then
    MEM_MB_EACH=$(expr $(free -m | grep -oP '\d+' | sed '6!d') / 2)
    CPUS_EACH=$(expr $(nproc --all) / 2)
    docker run --name "group-0" --rm -d --memory="$MEM_MB_EACH"m --cpus="$CPUS_EACH" -v $(readlink -f $SSH_AUTH_SOCK):/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent --runtime=nvidia --gpus '"device=0,1,2,3"' $MLAB2_IMAGE
    docker run --name "group-1" --rm -d --memory="$MEM_MB_EACH"m --cpus="$CPUS_EACH" -v $(readlink -f $SSH_AUTH_SOCK):/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent --runtime=nvidia --gpus '"device=4,5,6,7"' $MLAB2_IMAGE
fi

# Run the preload script
sleep 15
for container in $(docker ps -a -q); do
    docker exec -d $container /root/mlab2/.env/bin/python /root/mlab2/w3d1_preload.py
done
