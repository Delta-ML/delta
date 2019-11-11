# How to build the wheel file

## Intro

In order to provide users a simpler way to install `Delta`, we need to
build the Wheel file `.whl` and upload this wheel file to Pypi's
website. Once we uploaded the wheel file, all that users need to do is
typing `pip install delta-nlp`.

**Notice**: installation by pip only supports NLP tasks now. If you need the
full version of the Delta (with speech tasks), you should install the
platform from source.

## Prepare

For linux wheel building, you will need the docker image:

```bash
docker pull tensorflow/tensorflow:custom-op-ubuntu16
```

Before build the wheel file, you need to install the `DELTA` before. Run the Makefile script for NLP version.

```bash
cd tools && make nlp
```

## Start to build

### MacOS

```bash
bash ./tools/install/build_pip_pkg.sh
```

The generated wheel will be under `dist` like
`delta_nlp-{version}-cp36-cp36m-macosx_10_7_x86_64.whl`

### Linux

Wheel building in linux is more complicated. You need to run a docker 

```bash
docker run --name delta_pip_tf2_u16 -it -v $PWD:/delta  tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```

In the docker environment, run:

```bash
bash ./tools/install/build_pip_pkg.sh
```

The generated wheel will be under `dist` like
`delta_nlp-{version}-cp36-cp36m-linux_x86_64.whl`

Repair the wheel file for multiple linux platform support:

```bash
auditwheel repair dist/xxx.whl
```

The final wheel will be under `wheelhouse` like
`delta_nlp-{version}-cp36-cp36m-manylinux1_x86_64.whl`.

## Upload

After building the wheel file, upload these files to Pypi:

```
twine upload xxx.whl
```
