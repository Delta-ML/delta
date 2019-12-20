# Develop with Docker 

## Install Docker

Make sure `docker` has been installed. You can refer to the [official tutorial](https://docs.docker.com/install/).

## Development with Docker

You can build DETLA or DETLANN locally as [Build Images](#build-images),
or using pre-build images as belows:

All avaible image tags list in [here](https://cloud.docker.com/repository/docker/zh794390558/delta/tags),
please choose one as needed.

If we choose `delta-cpu-py3`, then download the image as below:

```bash
docker pull zh794390558/delta:delta-cpu-py3
```

After the image downloaded, create a container:

```bash
cd /path/to/detla && docker run -it -v $PWD:/delta zh794390558/delta:delta-cpu-py3 /bin/bash
```

then develop as usual. 

We recommend using a power machine to develop DELTANN, since it needs to compile
`Tensorflow` which is time-consuming.

### Tags
Please go to [this](https://hub.docker.com/r/zh794390558/delta/tags) to see the valid docker images tags.

## Build Images

### Build CI Image

```bash
pushd docker && bash build.sh ci cpu build && popd
```

### Build DELTA Image

For building cpu image:

```bash
pushd docker && bash build.sh delta cpu build && popd
```

for building gpu image

```bash
pushd docker && bash build.sh delta gpu build && popd
```

### Build DELTANN Image

For building cpu image:

```bash
pushd docker && bash build.sh deltann cpu build && popd
```

for building gpu image

```bash
pushd docker && bash build.sh deltann gpu build && popd
```
