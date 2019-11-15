# Installation using Docker

You can directly pull the pre-build docker images for DELTA and DELTANN. We have created the following docker images:

- delta-gpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:delta-gpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)

- delta-cpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:delta-cpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)

- deltann-gpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:deltann-gpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)

- deltann-cpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:deltann-cpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)



## Install Docker

Make sure `docker` has been installed. You can refer to the [official tutorial](https://docs.docker.com/install/).

## Pull Docker Image

You can build DETLA or DETLANN locally as [Build Images](#build-images),
or using pre-build images as belows:

All avaible image tags list in [here](https://cloud.docker.com/repository/docker/zh794390558/delta/tags),
please choose one as needed.

If you choose `delta-cpu-py3`, then download the image as below:

```bash
docker pull zh794390558/delta:delta-cpu-py3
```

## Create Container

After the image downloaded, create a container.

For **delta** usage (model development):

```bash
cd /path/to/detla && docker run -v `pwd`:/delta -it zh794390558/delta:delta-cpu-py3 /bin/bash
```

The basic version of **delta** (except Kaldi) was already installed in this container. You can develop in this container like:

```python
# Add DELTA enviornment
source env.sh

# Generate mock data for text classification.
pushd egs/mock_text_cls_data/text_cls/v1
./run.sh
popd

# Train the model
python3 delta/main.py --cmd train_and_eval --config egs/mock_text_cls_data/text_cls/v1/config/han-cls.yml
```

For **deltann** usage (model deployment):

```bash
cd /path/to/detla 
WORKSPACE=$PWD
docker run -it -v $WORKSPACE:$WORKSPACE zh794390558/delta:deltann-cpu-py3 /bin/bash
```

We recommend using a high-end machine to develop DELTANN, since it needs to compile
`Tensorflow` which is time-consuming.

