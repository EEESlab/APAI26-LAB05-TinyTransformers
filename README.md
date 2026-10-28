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

### Option B: Lab machines (virtual machine)

1. Open the VirtualBox virtual machine
2. Open a terminal (CTRL+T or open terminal)
3. Go into 'APAI-Docker' folder.
4. Open VSCode with `code .`
5. On VSCode, click on 'Reopen in container'.
6. Open a new terminal in VSCode and launch the following commands to clone this repository:
```
git clone https://github.com/EEESlab/APAI26-LAB05-TinyTransformers
cd APAI26-LAB05-TinyTransformers/
```
7. **[DO NOT FORGET]** Every time you open a new terminal run:
```
source setup-pulp-sdk.sh
```
8. Now you're ready to start!

## How to run the code

Run:

```
make clean all run
```

The first time, `make` also downloads [PULP-TrainLib](https://github.com/pulp-platform/pulp-trainlib) and prepares the lab (see `initialization.sh`).

To run on N cluster cores (1 to 8, default 8):

```
make clean all run NUM_CORES=N
```

To print the performance counters (TASK 2.2), uncomment the line `APP_CFLAGS += -DSTATS` in the `Makefile`.
