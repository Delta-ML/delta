# Intallation using Docker

You can directly pull the pre-build docker images for DELTA and DELTANN. We have created the following docker images:

- TF1.14.0-delta-gpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:1.14.0-delta-gpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)

- TF1.14.0-delta-cpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:1.14.0-delta-cpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)

- TF1.14.0-deltann-gpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:1.14.0-deltann-gpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)

- TF1.14.0-deltann-cpu-py3 [![](https://images.microbadger.com/badges/image/zh794390558/delta:1.14.0-deltann-cpu-py3.svg)](https://hub.docker.com/r/zh794390558/delta)



## Install Docker

Make sure `docker` has been installed. You can refer to the [official tutorial](https://docs.docker.com/install/).

## Pull Docker Image

You can build DETLA or DETLANN locally as [Build Images](#build-images),
or using pre-build images as belows:

All avaible image tags list in [here](https://cloud.docker.com/repository/docker/zh794390558/delta/tags),
please choose one as needed.

If you choose `1.14.0-delta-cpu-py3`, then download the image as below:

```bash
docker pull zh794390558/delta:1.14.0-delta-cpu-py3
```

## Create Container

After the image downloaded, create a container:

```bash
cd /path/to/detla && docker run -it -v $PWD:/delta zh794390558/delta:1.14.0-delta-cpu-py3 /bin/bash
```

then develop as usual. 

We recommend using a high-end machine to develop DELTANN, since it needs to compile
`Tensorflow` which is time-consuming.
