set -e

###### Install miniconda and other dependencies ######

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p
rm -rf Miniconda3-py39_4.12.0-Linux-x86_64.sh
PATH=$HOME/miniconda3/bin:$PATH
conda init

# Install dependencies for w3d2
sudo apt-get install swig cmake freeglut3-dev xvfb -y

###### Clone and install the repo ######

# Check for ssh key file
if [ ! -f ~/.ssh/mlab2_ssh ]; then
    echo "~/.ssh/mlab2_ssh does not exist on remote instance. Please ensure key is provided and re-run script."
    exit 1
fi

# Set correct permissions for the ssh key and add it to the ssh-agent
chmod 600 ~/.ssh/mlab2_ssh
eval `ssh-agent -s`
ssh-add ~/.ssh/mlab2_ssh

# Skip over prompt to add github to known_hosts
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Clone, create conda env, and install dependencies
ENV_PATH=~/mlab2/.env/
git clone git@github.com:mlab-account/mlab2.git
cd mlab2
conda create -p $ENV_PATH python=3.9 -y
conda install -p $ENV_PATH pytorch=1.11.0 torchtext torchdata torchvision cudatoolkit=11.3 -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
cp infrastructure/pre-commit .git/hooks/pre-commit # Set up pre-commit hook to protect main branch

# Install pycuda (needed for w1d6) and ffmpeg (needed for w3d2) 
export CUDA_HOME=/usr/lib/cuda
conda install -p $ENV_PATH ffmpeg pycuda -c conda-forge -y

###### Set up .bashrc and .gitconfig ######

# Add the ssh key to the ssh-agent each time
echo '
# Add github service account SSH key to agent
if [ -z "$SSH_AUTH_SOCK" ] ; then
  eval `ssh-agent -s`
  ssh-add ~/.ssh/mlab2_ssh
fi
' >> ~/.bashrc

# Install VSCode extensions if they aren't there
echo '
# Install VSCode extensions if needed
if which code &> /dev/null; then 
  extensions=$(code --list-extensions)
  if ! [[ $extensions == *"ms-python.python"* && $extensions == *"bierner.markdown-mermaid"* ]]; then
    echo "Installing VSCode extensions..."
    code --install-extension ms-python.python &> /dev/null
    code --install-extension bierner.markdown-mermaid &> /dev/null
  fi
fi
' >> ~/.bashrc

# Activate the mlab2 virtualenv each time
echo '
conda activate ~/mlab2/.env/
export PATH="/home/ubuntu/mlab2/.env/bin:$PATH"
' >> ~/.bashrc

# Set up .gitconfig
echo '[user]
        name = MLAB Account
        email = 110868426+mlab-account@users.noreply.github.com
' > ~/.gitconfig
