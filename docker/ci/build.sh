#!/bin/bash

if [ -z $MAIN_ROOT ];then
  pushd ../..; . env.sh; popd
fi

on_exit() {
  rm requirements.txt
  rm sources.list.ubuntu18.04
  rm $MAIN_ROOT/docker/ci/dockerfile
}
trap on_exit EXIT HUP PIPE QUIT TERM

DOCKER="sudo docker"
PIP_INSTALL="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/"
BASE_IMAGE=tensorflow/tensorflow:1.14.0-py3

# pull latest image
${DOCKER} pull $BASE_IMAGE

# generate dockerfile
cp ../sources.list.ubuntu18.04 .
cp $MAIN_ROOT/tools/requirements.txt .

cat > $MAIN_ROOT/docker/ci/dockerfile <<EOF
FROM $BASE_IMAGE
COPY sources.list.ubuntu18.04 /etc/apt/sources.list

# install tools
COPY install.sh /ci/install.sh
#RUN /bin/bash chmod +x /ci/install.sh && /bin/bash /ci/install.sh
RUN /bin/bash /ci/install.sh

# add users
RUN adduser --disabled-password --gecos '' gitlab-runner && adduser gitlab-runner sudo
# set sudoers
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER gitlab-runner

# install gcc
COPY install_user.sh /ci/install_user.sh
RUN /bin/bash /ci/install_user.sh

# install python pakcages
COPY requirements.txt /ci/requirements.txt
WORKDIR /ci
RUN $PIP_INSTALL --upgrade pip && $PIP_INSTALL --user -r requirements.txt

CMD ["/bin/bash", "-c"]
EOF


# build image
TARGET=ci
DEVICE=cpu
TAG=1.14.0-${TARGET}-${DEVICE}-py3
${DOCKER} build --no-cache=false -t delta:$TAG -f $MAIN_ROOT/docker/ci/dockerfile . || { echo "build ci error"; exit 1; }

# push image
${DOCKER} tag delta:${TAG} zh794390558/delta:${TAG}
${DOCKER} push zh794390558/delta:${TAG}

if [ $? == 0 ]
then
  echo "push successful";
  exit 0
else
  echo "push error";
  echo "make sure you have login to docker"
  echo "please run 'docker login'"
  exit 10
fi
