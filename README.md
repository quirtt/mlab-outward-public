# MLAB Participant Instructions

## Course Outline

- W0D1 - pre-course exercises on PyTorch and einops (CPU)
- W1D1 - practice PyTorch by building a simple raytracer (CPU)
- W1D2 - build your own ResNet (GPU preferred)
- W1D3 - build your own backpropagation framework (CPU)
- W1D4 - model training
  - Part 1: model training and optimizers (CPU)
  - Part 2: hyperparameter search (GPU preferred)
- W1D5 - cumulants and polysemanticity (CPU)
- W1D6 - GPU programming with PyCuda (GPU required)
- W2D1 - build your own BERT (CPU)
- W2D2 - BERT
  - Part 1: data preparation for sentiment analysis (CPU)
  - Part 2: fine-tuning BERT on sentiment analysis (CPU)
  - Part 3: data preparation for masked language modelling (CPU)
  - Part 4: training BERT from scratch (GPU required)
- W2D3 - GPT
  - Part 1: build your own GPT (CPU)
  - Part 2: sampling text from GPT (GPU preferred)
- W2D4 - transformer interpretability (CPU)
- W2D5 - transformer interpretability on algorithmic tasks (CPU)
- W3D1 - distributed training
  - Part 1: torch.distributed (GPU required)
  - Part 2: data parallelism (GPU preferred)
  - Part 3: tensor parallelism (GPU preferred)
- W3D2 - intro to RL
  - Part 1: multi-armed bandit (CPU)
  - Part 2: DQN (CPU)
- W3D3 - policy gradients and PPO (CPU)
- W3D4 - diffusion models
  - Part 1: introduction to diffusion (CPU)
  - Part 2: U-Net architecture (GPU preferred)
  - Part 3: training a diffusion model on FashionMNIST (GPU required)
- W3D5
  - Part 1: CLIP (CPU)
  - Part 2: image generation with Stable Diffusion (GPU required)

For days marked "CPU", you can complete all the learning objectives with only a CPU, and a GPU may or may not provide speedups. "GPU preferred" means that you can still learn everything on CPU, but you might have to wait for things to run and/or use smaller model sizes. "GPU required" means without a GPU, it will be very tedious or impossible to complete all the content. 

Note that W1D5 and W2D5 are not publicly available in the mlab-outward repo because they rely on unpublished work at Redwood Research. 

## Self-Study Instructions

Read this section if you're working on MLAB2 independently using your own hardware, or using a remote machine over SSH. If you're really stuck getting set up, email me at chris (at) rdwrs.com and I'll do my best to troubleshoot.

### Local Installation

Working locally can be a good option as it's simpler than setting up a remote machine, and for most of the content you don't need a GPU. The course has been tested on Windows 10, macOS, and Ubuntu. To set up your machine, the simplest way is to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and then create a virtual environment containing the correct versions of the dependencies. On macOS or Linux you can do this (setting ENV_PATH to the absolute path to your cloned repo):

```
ENV_PATH=~/mlab2/.env/
cd $ENV_PATH
conda create -p $ENV_PATH python=3.9 -y
conda install -p $ENV_PATH pytorch=1.11.0 torchtext torchdata torchvision cudatoolkit=11.3 -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
```

On Windows, the equivalent is:

```
$env:ENV_PATH ='c:\users\chris\mlab2\.env'
cd mlab2
conda create -p $env:ENV_PATH python=3.9 -y
conda install -p $env:ENV_PATH pytorch=1.11.0 torchtext torchdata torchvision cudatoolkit=11.3 -c pytorch -y
conda run -p $env:ENV_PATH pip install -r requirements.txt
```

If you're using Visual Studio Code, we use the Python extension as well as an extension called Mermaid to render diagrams in Markdown. You can install these by hand in the VSCode editor, or from the command line like this:

```
code --install-extension ms-python.python
code --install-extension bierner.markdown-mermaid
```

At this point, if you open the repo folder in Visual Studio Code, in the bottom right it should recognize the virtual environment - something like "Python 3.9.13 (conda)". If you don't see this, try opening a `.py` file and if you still don't see it, the Python extension might not be installed properly. 

A couple of the days have additional dependencies - I recommend skipping these until you reach those days.

W1D6: You need PyCuda: `conda install -p $ENV_PATH pycuda -c conda-forge -y`
W3D2: To see videos of your RL agent playing, you need ffmpeg `conda install -p $ENV_PATH ffmpeg -c conda-forge -y`

### Remote Installation

You can also run over SSH on a remote machine with a GPU. Some options I've used are AWS and Lambda Labs. For AWS, I used a `p3.2xlarge` instance with the `Deep Learning AMI GPU PyTorch 1.11.0` AMI. You'll need to create your own SSH key (Google if you're not sure how to do this on your operating system), then follow the section "In-Person Instructions" below to set up Visual Studio Code to connect to the remote machine over SSH.


## In-Person Instructions

Welcome to MLAB! Many of the exercises we have planned require access to a GPU, so we'll provide a remote GPU instance for you to use each day. You'll connect to it with the VSCode remote ssh extension, which should make the experience a lot like coding on your local machine. We'll also have you use git to save your work each day.

There are a few setup steps that you'll need to do. Some of these are one-time things at the start of the program, others will need to be done each day.

## First time setup

### SSH Key - Linux or Mac

- Download the mlab2_ssh key file [from Slack](https://mlab-2.slack.com/archives/C03NNTX16NS/p1660581218286919).
- Make an ssh directory (if you don't already have one): `mkdir -p ~/.ssh`
- Copy the downloaded key to the directory: `cp ~/Downloads/mlab2_ssh ~/.ssh/mlab2_ssh`
- Set permissions: `chmod 600 ~/.ssh/mlab2_ssh`


### SSH Key - Windows

- Download the mlab2_ssh key file [from Slack](https://mlab-2.slack.com/archives/C03NNTX16NS/p1660581218286919).
- Make an ssh directory (if you don't already have one): `md -Force ~\.ssh`
- If that doesn't work, try `md -Force C:\Users\[your user name]\.ssh`
- If that doesn't work for you, confirm that this folder already exists for yourself, and if not make it in the file navigator application.
- Copy the downloaded key to the directory.
- [Set permissions on the key](https://superuser.com/a/1296046)
- If you still see permission denied errors, make sure that your mlab2_ssh file is using LF line endings and has a final line break at the end (so there shouldn't be any text on the last line of the file).

### Install the VSCode remote-ssh extension

- Have VSCode installed
- Follow [this link](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and click "Install"

### Set up your ssh config file

- Click the button in the bottom left of VSCode, which brings up the remote ssh options menu.
- Choose "Open SSH Configuration File...", and if it prompts you to choose a file pick the first one.
- Paste in the following and then save the file.
    - If there's already stuff in the file, put this at the end.
    - If you have an IP address for your pair's GPU instance, replace `<instance ip address>` with that IP address.

```text
Host mlab2
  HostName <instance ip address>
  User ubuntu
  IdentityFile ~/.ssh/mlab2_ssh
  StrictHostKeyChecking no
```

## Everyday Instructions

Connect to your GPU instance

- Get your pair's instance IP address from the pairs sheet.
- Set the IP address in your ssh config file: open the config file, edit the HostName line with the IP address for the day, and save the file.
- Click the  button in the bottom left, choose "Connect to Host...", and choose "mlab2"
- Click the file button in the top left, click "Open Folder", choose the "mlab2" directory, and hit "OK"
- If it asks you if you trust the authors, click "Yes"

### Using git

Each day you should make a new branch off of main, make commits to it and push it up to the repo. This is how your work during MLAB will be saved for you to reference later.

Make a branch for the day: `git checkout -b <branch name>`, where your branch name should follow the convention `w#d#/<name>-and-<name>`.

For example, if Tamera and Edmund were pairing on the week 1 day 3 content, the command would be `git checkout -b w1d3/tamera-and-edmund`
If you share a first name with someone else in the program, you should probably include your last initial or last name in your branch name for disambiguation. Create a new file for your answers and work through the material with your partner.

To view the instructions for a day, right-click the `w#d#_instructions.md` file in the file menu on the left, and click "Open Preview"
The recommended way to do the exercises is to make a new `.py` file (suggested name: `w#d#_answers.py`) and start it with the characters `# %%`.

Then paste in a block of code from the instructions doc and add another `# %%` line. In VSCode, if you have the Python extension installed, you should see the option to "Run Cell" at the top of the file; this will start an interactive session you can use to run the code. If you add more code at the bottom of the file and follow it with another `# %%`, this will create another cell which can be run independently in the same session. Cells can be run many times and in any order you choose; the session will maintain variables and state until it is restarted. Add code cells and run them, filling in your answers as you go, to carry out the exercises.
As you work, commit changes to your branch and push them to the repo.

To make a commit:

```bash
git add :/
git commit -m '<your commit message>'
```

To push your commits:

- The first time pushing your branch: `git push -u origin <branch name>` (or run git push and copy and paste the suggested command)
- Every other time: `git push`
- Make sure to commit and push at least once at the end of the day! We'll reset the repo on all instances each night, so if you haven't pushed your work to GitHub it will be lost.