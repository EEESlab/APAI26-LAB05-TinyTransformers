# APAI-LAB05: Tiny Transformers

## Material

Assignment: [here](_docs/assignment.docx)
Slides: [here](_docs/slides.pdf)


## How to deliver the assignment:

Use Virtuale, upload only the assignment file named as follows: LAB#_APAI_yourname.pdf


**Assignment DEADLINE: Sunday 15/11/2026 (end of day, 23:59)**

___

## Quickstart

You are encouraged to work on your own laptop using the provided devcontainer (Option A). If that is not possible, you can use the virtual machine available on the lab machines (Option B).

### Option A: Your own laptop (devcontainer, recommended)

Requirements: [Docker](https://docs.docker.com/get-docker/), [VSCode](https://code.visualstudio.com/) and the VSCode [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

1. Open a terminal and clone this repository:
```
git clone https://github.com/EEESlab/APAI26-LAB05-TinyTransformers
cd APAI26-LAB05-TinyTransformers/
```
2. Open VSCode with `code .`
3. On VSCode, click on 'Reopen in container' (the first time, the container image is downloaded, which may take a while).
4. Open a new terminal in VSCode: the PULP SDK is configured automatically.
5. Now you're ready to start!

On Windows, install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) first, and choose the Windows version of Docker Desktop (even if you work with WSL). If VSCode reports that Docker cannot be found, enable the WSL integration in Docker Desktop (Settings -> Resources -> WSL integration), then Apply & Restart.

### Option B: Lab machines (virtual machine)

The Ubuntu-based virtual machine already contains the PULP SDK and VSCode (used only to edit the code and open terminals), so no Docker container is needed.

1. From a browser **outside** the virtual machine, download the lab zip file from Virtuale (the only website reachable from the lab machines) and extract it (right-click -> Extract).
2. Go to `C:/VM_APAI` and double-click `PULP-box.ova`. When VirtualBox opens, click on "Finish"/"Fine".
3. Once the virtual machine is imported, start it with "Start"/"Avvia".
4. Log in with username `pulp-box` and password `pulp`.
5. From the VirtualBox top bar, open Machine -> File Manager (left panel: files of the PC, right panel: files of the VM). Enter the login credentials at the bottom of the right panel.
6. Drag and drop the extracted lab folder into a subdirectory of the VM home directory (you need write rights there).
7. Inside the VM, open a terminal and go to the lab folder with `cd`.
8. **[DO NOT FORGET]** Every time you open a new terminal run:
```
source setup_pulp_sdk.sh
```
9. Now you're ready to start!

## How to run the code

Run:

```
make clean all run
```

The first time, `make` also downloads [PULP-TrainLib](https://github.com/pulp-platform/pulp-trainlib) and prepares the lab (see `initialization.sh`). The zip file on Virtuale already includes PULP-TrainLib.

To run on N cluster cores (1 to 8, default 8):

```
make clean all run NUM_CORES=N
```

To print the performance counters (TASK 2.2), uncomment the line `APP_CFLAGS += -DSTATS` in the `Makefile`. Comment it again for TASK 3, otherwise the story is mixed with the counters.
