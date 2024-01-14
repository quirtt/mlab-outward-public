# MLAB August 2022 - Staff Directions

## Setup

- Requires Python 3.9 or higher and PyTorch 1.11. PyTorch 1.12 will likely work but is not officially supported.
- Use of a conda environment or at least a virtualenv is highly recommended.
- Install `torch` and related dependencies with `conda install pytorch=1.11.0 torchdata -c pytorch -y`
- Install other dependencies with: `pip install -r requirements.txt`. Note if you want to run the optional CUDA day, additional dependencies are required. If a needed dependency is not in the requirements, please message me or file a PR.
- Install the extension [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint), but don't worry much about the warnings it gives at this point.
- Optionally, install the VSCode extension [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) to see diagrams generated with Mermaid.

### PyCUDA Setup

One of the optional days is about compiling your own CUDA kernels. On Windows this has a few steps.

- First, run `conda install -c conda-forge cudatoolkit-dev` to install a script which in turn will automatically download the CUDA developer tools from NVIDIA. After this step, you should be able to run `nvcc  --version` and see what version of the CUDA compiler you have. I had 11.4.
- You will need a compatible C/C++ compiler. On Windows, this means [consulting the following table](https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html) and downloading the matching version of Visual Studio Community (which was 2017 for me).
- The compiler `cl.exe` needs to be on your PATH. You can do this within the Python script itself - see `w1d6_solution.py` for where it was on my system. Note you may have multiple copies of `cl.exe` on your system - if your `cl.exe` gives compiler errors about `size_t` then you may be pointing at a 32-bit version instead of a 64-bit version.
- Now `conda install pycuda` should work.
- `CUDA_PATH` environment variable also needs to be set to point to a folder that has `bin/nvcc`. I did this inline in my script - see `w1d6_solution.py` again. On my system this was `C:/Users/Chris/miniconda3/Library/`.

### RL Setup

To record videos of agents playing in environments like CartPole, you need `ffmpeg`. On the remote machines, you also need `xvfb` to create a virtual display that Gym will render onto.

On Linux:

- `apt-get install swig cmake ffmpeg freeglut3-dev xvfb`

On Windows, download an [executable of FFMPEG](https://ffmpeg.org/download.html) and make sure it's on your system path (a restart may be required).

It's possible to save MP4 video of the agents playing as well. On a remote machine, install [this extension](https://marketplace.visualstudio.com/items?itemName=analytic-signal.preview-mp4) to be able to watch these MP4s within Visual Studio.

### Curriculum Authoring Setup

- Open a terminal and run `python build_all.py`. It should automatically re-generated the corresponding Markdown file containing student directions when a Python script ending in `_solution.py` is saved.

- Don't edit the Markdown file directly - to help you these are marked as read-only so you'll get a warning when you try to save edits.

- To automatically build instructions when committing, install pre-commit using `pip install pre-commit`, then run `pre-commit install`.

## Guide to Writing Directions

- Directions and solutions are versioned together in a `*_solution.py` script.
- Running the script should run all the tests against the solution.
- An if statement with the literal string "SOLUTION" as the condition like `if "SOLUTION":` means that the contents of the block will be omitted from the instructions. The else block will be included instead, or a placeholder "TODO" if there is no else block.
- An if statement with the literal string "SKIP" as the condition like `if "SKIP"` omits the contents of the block as well.
- Function or method definitions containing "SOLUTION" in their body replace that expression and any remaining body with a pass statement. This avoids losing an additional indentation level.
- Spoilers can be hidden by default using `<details>` and `<summary>` tags as shown in the below example.
- Equations can be written inline using dollar signs as delimiters like `$ /sqrt{2} $`. Note that you should use raw strings for the blocks containing equations (a raw string has the character `r` before the opening quotation mark), since this will ensure proper handling of `\` characters.
- Diagrams can be written using Mermaid notation. Docs and interactive editor are [here](https://mermaid.live/).
- The solution script should be importable without doing anything time-consuming or printing output, because some unit tests import the solution script to check against. Define `MAIN = __name__ == "__main__"` and then guard blocks with `if MAIN:`
- Avoid placing Python scripts in folders, as I don't want to require students to understand how relative imports work or have to `pip install -e` the repo. This was confusing and error prone in MLAB1. Code can assume that the working directory is always the root of the repo.
- Please don't commit any files larger than a couple megabytes, as this slows down cloning the repo for everyone. We can use Redwood's shared filesystem to store model weights, input text, or whatever.
- If at all possible, unit tests should not require a GPU and should take no longer than 1 second to run.
- Type hints are preferred, but optional. In some cases, upstream code doesn't have correct type hints; in this case either manually use `typing.cast(ActualType, obj)` or do "# type: ignore" and leave a comment with the problem.
- Comments using `#` in the solution file are NOT included in the instructions file, so you can use these for comments intended only for solution readers or other TAs.

### Remote Development Setup

One way to do is it set up the Host in your `~/.ssh/config` like this:

```text
Host rr
    HostName 18.237.88.205
    User ubuntu
    IdentityFile C:/users/chris/.ssh/ChrisMPem.pem
```

Your IdentityFile should be a SSH private key. It may have a different name or extension like `id_rsa`.

In VS Code, install the [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension. Then from the command palette, select "Remote-SSH: Connect to Host". It should find the host from your `config` and allow you to select it as an option.

If this isn't working, try the [official instructions](https://code.visualstudio.com/docs/remote/ssh); if your question isn't answered there then ask in the Slack.

In the bottom bar of VS Code you should see the current `conda` environment - make sure the correct one is selected.

On the remote, you won't be able to clone over SSH until you create a private key using `ssh-keygen` and add the corresponding public key to your GitHub account. You'll also need to set your git email and name like this:

```text
git config --global user.email "chrism6794@gmail.com"
git config --global user.name "Chris MacLeod"
```

## Example Spoiler

<details>
<summary>Click on this to show spoiler</summary>

### Spoiler Contents

This is the contents of the spoiler. Remember to put an empty line after the summary end tag to make Markdown work inside the spoiler properly.

</details>

## Example Equations

Inline equations with single dollar sign: $\sqrt{2}$

Equation on separate line with double dollar signs:

$$ x=\frac{-b \pm \sqrt{b^2-4c}}{2} $$

## Example Diagram

```mermaid
graph TD  %% top down
    Input --> LayerNorm1[Layer Norm 1<br>Attention] --> Residual1[Add] --> LayerNorm2[Layer Norm 2<br>MLP]--> Residual2 --> BlOutput[Output]
    Input --> Residual1
    Residual1 --> Residual2[Add]
```

## Known Issues

- VSCode bug: The Markdown preview will only pick up changes to the instructions.md if you also have the instructions.md opened in its own tab.

- The build script doesn't like Unicode characters like smart quotes. You can search your code by regex for `[^\x00-\x7f]` to find non-ASCII characters, or run `fix_special_characters.py solution_file.py` to automatically replace them.

*If you have mismatched opening and closing HTML tags, VSCode will just refuse to load the preview without giving any kind of error message. On GitHub you can view the rendered Markdown file to troubleshoot the issue.

*GitHub tries to render the equations, but it has bugs right now. VS Code should render them properly.

- If you see "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.", try reinstalling NumPy.