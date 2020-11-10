#!/bin/bash 

if [ $# != 3 ];then
  echo "usage: $0 [ci|delta|deltann] [cpu|gpu] [build|push|dockerfile]"
  exit 1
fi

if [ -z $MAIN_ROOT ];then
  # TF_VER, PY_VER
  pushd .. && . env.sh && popd
fi

# generate dockerfile
TARGET=$1 #ci, delta or deltann
DEVICE=$2 # cpu or gpu
MODE=$3 # push image or generate dockerfile

ADD_USER=false # false for root user
INSTALL_GCC48=false  # install gcc-4.8
if [ $TARGET == 'deltann' ];then
  INSTALL_GCC48=false
fi
SAVE_DOCKERFILE=false
if [ $MODE == 'dockerfile' ];then
  SAVE_DOCKERFILE=true
fi


# docker file
DOCKERFILE=$MAIN_ROOT/docker/dockerfile.${TARGET}.${DEVICE} 
if [ -f $DOCKERFILE ];then
  unlink $DOCKERFILE
fi

TAG=${TF_VER}-${TARGET}-${DEVICE}-py3
DOCKER='sudo docker'
PIP_INSTALL="pip --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple"

set -e
set -u
set -o pipefail

on_exit() {
  if [ $SAVE_DOCKERFILE == false ];then
    rm $DOCKERFILE 
  fi
}
trap on_exit HUP INT PIPE QUIT TERM EXIT


# https://hub.docker.com/r/tensorflow/tensorflow
# Versioned images <= 1.15.0 (1.x) and <= 2.1.0 (2.x) have Python 3 
# (3.5 for Ubuntu 16-based images; 3.6 for Ubuntu 18-based images) in images tagged "-py3"
# and Python 2.7 in images without "py" in the tag. 
# All newer images are Python 3 only. Tags containing -py3 are deprecated.
if [ ${DEVICE} == 'cpu' ] && [ ${TARGET} == 'deltann' ];then
  IMAGE=tensorflow/tensorflow:devel
elif [ ${DEVICE} == 'gpu' ] && [ ${TARGET} == 'deltann' ];then
  IMAGE=tensorflow/tensorflow:devel-gpu
elif [ ${DEVICE} == 'cpu' ] && [ ${TARGET} == 'delta' ] || [ ${TARGET} == 'ci' ];then
  IMAGE=tensorflow/tensorflow:${TF_VER}
elif [ ${DEVICE} == 'gpu' ] && [ ${TARGET} == 'delta' ];then
  IMAGE=tensorflow/tensorflow:${TF_VER}-gpu
else
  echo "no support target or device"
  exit -1
fi

# generate dockerfile
cp $MAIN_ROOT/tools/requirements.txt .


# source images
cat > $DOCKERFILE <<EOF
FROM ${IMAGE}
COPY sources.list.ubuntu18.04 /etc/apt/sources.list

# install tools 
COPY install.sh /install.sh
RUN /bin/bash /install.sh

EOF

# add user
if [ $ADD_USER == true ];then
cat >> $DOCKERFILE <<EOF
#add user
ENV ROLE ${TARGET}
EOF

cat >> $DOCKERFILE <<EOF
RUN adduser --disabled-password --gecos '' --shell '/bin/bash' \$ROLE \
  && adduser \$ROLE sudo \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER \$ROLE 

EOF
fi # add user


# install gcc
if [ ${INSTALL_GCC48} == true ]; then
cat >> $DOCKERFILE <<EOF
# install gcc
COPY install_user.sh /ci/install_user.sh
RUN /bin/bash /ci/install_user.sh

EOF
fi

if [ ${TARGET} == 'ci' ];then
# install python pakcages
cat >> $DOCKERFILE <<EOF
COPY requirements.txt /ci/requirements.txt
WORKDIR /ci
RUN sudo $PIP_INSTALL --upgrade pip && $PIP_INSTALL --user -r requirements.txt

EOF

elif [ ${TARGET} == 'delta' ];then
cat >> $DOCKERFILE <<EOF
RUN sudo mkdir workspace
RUN cd /workspace && git clone --depth 1 https://github.com/didi/delta.git
RUN cd /workspace/delta/tools && make basic
WORKDIR /workspace/delta

EOF

fi

cat >> $DOCKERFILE <<EOF
CMD ["/bin/bash", "-c"]
EOF


if [ $MODE == 'push' ] || [ $MODE == 'build' ];then
  # pull latest image
  $DOCKER pull $IMAGE
  
  # build image
  $DOCKER build --no-cache=false -t zh794390558/delta:$TAG -f $DOCKERFILE . || { echo "build ${TARGET} ${DEVICE} error"; exit 1; }
  
  #push image
  if [ $MODE == 'push' ];then
    #$DOCKER tag delta:${TAG} zh794390558/delta:${TAG}
    $DOCKER push zh794390558/delta:$TAG

    if [ $? == 0  ]; then
        echo "push successful";
        exit 0
    else
        echo "push error";
        echo "make sure you have login docker"
        echo "plaese run 'docker login'"
        exit 10
    fi
  fi
fi

if [ -f $DOCKERFILE ] && [ ${SAVE_DOCKERFILE} == false ];then
  unlink $DOCKERFILE 
fi

