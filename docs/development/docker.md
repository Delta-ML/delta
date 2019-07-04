# Docker 

## [Install Docker](https://docs.docker.com/docker-for-mac/install/)

## Development with Docker

You can build DETLA or DETLANN locally as [Build Images](#build-images),
or using pre-build images as belows:

All aviable image tags list in [here](https://cloud.docker.com/repository/docker/zh794390558/delta/tags),
please choose one as needed.

If we choice `latest-devl-delta-cpu-py3`, then download the image as below:
```bash
docker pull zh794390558/delta:latest-devel-delta-cpu-py3
```

After image has beed downloaded, creating a contianer:

```bash
cd /path/to/detla && docker run -it -v $PWD:/delta zh794390558/delta:latest-devel-delta-cpu-py3 /bin/bash
```

then develop as usual. 

We recommend using a power machine to devleop DELTANN, since it needs to compile
`Tensorflow` which is time-consuming.


## Build Images

### Build CI Image

```bash
pushd docker/ci/ && bash build.sh && popd
```

### Build DELTA Image

For building devel cpu image:
```bash
pushd docker/devel/ && bash build.sh delta cpu && popd
```
for building devel gpu image

```bash
pushd docker/devel/ && bash build.sh delta gpu && popd
```

### Build DELTANN Image

For building devel cpu image:
```bash
pushd docker/devel/ && bash build.sh deltann cpu && popd
```
for building devel gpu image

```bash
pushd docker/devel/ && bash build.sh deltann gpu && popd
```


