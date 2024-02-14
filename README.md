# Simple Phi-2

Phi-2 is a very [accessible and powerful model](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/). This code will allow you to stand up the model (serve_phi2.py) as well as a chat interface (simple_chat.py) for you to interact with it. The model server is implemented using [Python's FastAPI](https://fastapi.tiangolo.com/). The user interface uses [Gradio](https://www.gradio.app/docs/interface). Despite Phi-2 being a small language model (SLM) it is recommended to run the model server on a CUDA compatible GPU with support for CUDA 12.0.1 or above.


## Quickstart

To get started quickly, you can simply use the pre-build containers. The commands below assume you have
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 
installed on your host. Please adjust `MODEL_ENDPOINT` according to your runtime setup.

```bash
docker run -d --gpus all -p 8080:8080 simonj.azurecr.io/server-phi2:latest
docker run -p 8881:8881 -e MODEL_ENDPOINT=http://localhost:8080  simonj.azurecr.io/phi2-ui:latest
```

Then, open your web browser and navigate to `http://localhost:8088` to interact with the chat interface.

## Building the Docker Images

If you want to build the Docker images yourself, you can do so using the provided Dockerfiles. Here's how you can build the model server image:

> **Note:** If your host's CUDA version doesn't match with the above you can adjust the base image in `Dockerfile.model` based
> on the [available runtime images listed here](https://hub.docker.com/r/nvidia/cuda/tags).

```bash
docker build -f Dockerfile.model -t model_server_image .
```

And here's how you can build the UI server image:

```bash
docker build -f Dockerfile.ui -t ui_image .
```