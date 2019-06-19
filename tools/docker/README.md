# Dockerfiles

This folder contains multiple `Dockerfiles` for an `Ubuntu 18.04` `base` and `vnc` images as well as the `minerl-gpu` image.

## `minerl-gpu` image

Ubuntu `18.04` container with `Python 3.6` that comes with [MineRL](https://github.com/minerllabs/minerl), the latest CUDA enabled PyTorch as well as [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/). Make sure you have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed to if you want to use it with a GPU.

### Setup

You have to build the containers yourself for now but we will look into setting up a container registry with the GitLab instance.

Clone the repository navigate to the `tools/docker` folder and build the containers via the provided `Makefile`.

```bash
git https://git.bioinf.jku.at/minerl/minerl-neurips-2019
pushd minerl-neurips-2019
pushd tools/docker
# Build the minerl-gpu container
make minerl-gpu
```

### Running the container

After building the container can either run directly via or via `docker-compose` (recommended).

```bash
docker run --runtime=nvidia -it -p 5901:5901 -p 8888:8888 -e VNC_PW=vncpassword ml-jku/minerl-gpu
```

### Docker Compose

Aside from using Docker directly the probably simplest way to launch a container is via [Docker Compose](https://docs.docker.com/compose/install/).

Navigate to the top of the folder structure and run:

```bash
docker-compose up
```

Running above command will automatically run a container based on the `ml-jku/minerl-gpu` image using the configuration specified in 'docker-compose.yml'.

_It is recommended to set a custom password instead of always copying the tokens._

#### Setup Password for JupyterLab

Create a personal version of `jupyter_notebook_config.py` in [resources](./resources) (see the [template](./resources/jupyter_notebook_config_template.py)) and set a custom password as described [here](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#preparing-a-hashed-password).
The configuration file will then be automatically mounted at startup and JupyterLab will use your specified settings.

#### Setup ports for the Docker container

You can specify your personal port mappings via an `.env` file. Create a new `.env` file:

```text
CONTAINER_NAME=ml-jku-minerl-gpu
VNC_PW=vncpassword
PTVSD=5678
VNC=5901
JUPYTER=8888
```

and update the name and the ports (i.e. your reserved ports from the **GPU Status** ports list).
Docker compose will automatically use your specified ports for the mappings.

#### Remote debugging

We show here how to setup remote debugging with [VSCode](https://code.visualstudio.com/).

Setup a new Python environment via `conda` or `virtualenv` and install the `ptvsd` package locally:

```bash
pip install --upgrade ptvsd
```

_Note_ You'll need to install `ptvsd` on both your local and the remote machine. The docker container comes already with `ptvsd` installed.

Launch VSCode in this folder and select the environment as your Python Interpreter. Open the file `ptvsd_debugging.py` locally, navigate to the debugger tab and create a new configuration (top left --> Add Configuration ...). Select _Remote Attach_ and enter hostname and port (e.g. localhost and the `PTVSD` port you set in `.env`).

VSCode will create a new file `.vscode/launch.json` that should look somewhat like this:

```JSON
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "port": 5678,
            "host": "localhost",
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/home/ml"
                }
            ]
        }
    ]
}
```

Make sure `remoteRoot` is set to the path where you mount your code (e.g. `/home/ml` for our example).

You can now launch the container via `docker-compose`

```bash
docker-compose up
```

The container will start and serve JupyterLab by default. Open a new terminal, copy the example to the container, then connect to the container:

```bash
# make sure to use the actual CONTAINER_NAME you specified in your .env file
docker cp examples CONTAINER_NAME:/home/ml/examples
docker exec -ti CONTAINER_NAME /bin/bash
```

Now you can start your code using the debugger via

```bash
python3 -m ptvsd --host 0.0.0.0 --port 5678 --wait examples/ptvsd_debugging.py
```

The Python process in the container will now wait until you attach to it. Switch to `ptvsd_debugging.py`, set some breakpoints and launch the debugger session.
If you encounter any problems create please create an issue or message Peter on Slack.

**IMPORTANT** The files on your local machine and the Docker container need to be identical and the folder structure has to match!
