# To run this image locally, run these commands and then use the VSCode remote extension to 
# "Attach to running container...":

# docker login --username=mlabredwood --password=dckr_pat_P-ma_kUDCWwDka8W7K-ta8iGbUE
# docker run --name mlab2 --rm -d --platform=linux/amd64 mlabredwood/mlab2:latest

# Or you can get jsut a terminal in the container with this command:

# docker exec -it mlab2 bash

# And then to shut it down when you're done:

# docker kill mlab2

# ------------------------------------------------------------------------------------------

# If you need the weights for the model Neel trained for w2d4, you can run this:

# docker login --username=mlabredwood --password=dckr_pat_P-ma_kUDCWwDka8W7K-ta8iGbUE
# docker run --name mlab2 --rm -d --platform=linux/amd64 mlabredwood/mlab2:1.1
# docker cp mlab2:/root/mlab2/w2d4_attn_only_2L.pth w2d4_attn_only_2L.pth
# docker kill mlab2

# ------------------------------------------------------------------------------------------

# You should be able to build the container if you have ssh access to the mlab2 repo 
# (takes 20-30 mins to run):

# DOCKER_BUILDKIT=1 docker build --ssh default -t mlab2 --platform linux/amd64 -f infrastructure/Dockerfile .

# ------------------------------------------------------------------------------------------

# This Docker image is hosted on a private Docker Hub repo as mlabredwood/mlab2. If you want
# to push new images to this repo, you'll need greater access to the mlabredwood Docker Hub
# account than the access token above will give you. The email address for the account is
# mlab@rdwrs.com, and Max Nadeau has the password for it if needed.

# After you've built the image, you'll need Docker Hub write access to push it:

# TAG="1.4" 
# docker tag mlab2 mlabredwood/mlab2:$TAG 
# docker push mlabredwood/mlab2:$TAG 
# docker tag mlab2 mlabredwood/mlab2:latest
# docker push mlabredwood/mlab2:latest

# All these commands were written on an M1 mac - there may be differences on other platforms.

# ------------------------------------------------------------------------------------------

FROM continuumio/miniconda3

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install wget git -y

# Install dependencies for w3d2
RUN DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles apt-get install swig cmake freeglut3-dev xvfb -y

# Clone the repo
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh cd $HOME && git clone git@github.com:mlab-account/mlab2.git

# Install the repo dependencies
ARG MLAB2_PATH=/root/mlab2 ENV_PATH=/root/mlab2/.env/
RUN conda init
RUN conda create -p $ENV_PATH python=3.9 -y
RUN conda install -p $ENV_PATH pytorch=1.11.0 torchtext torchdata torchvision cudatoolkit=11.3 -c pytorch -y
RUN conda run -p $ENV_PATH pip install -r $MLAB2_PATH/requirements.txt

# Set up pre-commit hook to protect main branch
RUN cp $MLAB2_PATH/infrastructure/pre-commit $MLAB2_PATH/.git/hooks/pre-commit

# Install pycuda (needed for w1d6) and ffmpeg (needed for w3d2)
RUN conda install -p $ENV_PATH ffmpeg pycuda -c conda-forge -y

# Copy model weights for w2d4
COPY w2d4_attn_only_2L.pth $MLAB2_PATH/w2d4_attn_only_2L.pth

# Install VSCode extensions if they aren't there
RUN echo '\n\
if which code &> /dev/null; then \n\
  extensions=$(code --list-extensions) \n\
  if ! [[ $extensions == *"ms-python.python"* && $extensions == *"bierner.markdown-mermaid"* ]]; then \n\
    echo "Installing VSCode extensions..." \n\
    code --install-extension ms-python.python &> /dev/null \n\
    code --install-extension bierner.markdown-mermaid &> /dev/null \n\
  fi \n\
fi \n\
' >> ~/.bashrc

# Activate the mlab2 virtualenv each time
RUN echo 'conda activate ~/mlab2/.env/\nexport PATH="/home/ubuntu/mlab2/.env/bin:$PATH"' >> ~/.bashrc

# Set up .gitconfig
RUN echo '[user]\n\
        name = MLAB Account\n\
        email = 110868426+mlab-account@users.noreply.github.com\n\
' > ~/.gitconfig

# Skip over prompt to add github to known_hosts
RUN ssh-keyscan -H github.com >> ~/.ssh/known_hosts

WORKDIR $MLAB2_PATH
CMD /bin/sh -c 'git fetch -p; git pull; sleep infinity'