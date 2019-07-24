#!/bin/bash 

if [ $# != 2 ];then
  echo "usage: $0 [delta|deltann] [cpu|gpu]"
  exit 1
fi

if [ -z $MAIN_ROOT ];then
  pushd ../..; . env.sh; popd
fi

if [ -f dockerfile ];then
  unlink dockerfile
fi

on_exit() {
  rm sources.list.ubuntu18.04
}
trap on_exit HUP INT PIPE QUIT TERM EXIT

# generate dockerfile
TARGET=$1 #delta or deltann
DEVICE=$2 # cpu or gpu
SAVE_DOCKERFILE=false
TYPE=devel
TAG=1.14.0-${TYPE}-${TARGET}-${DEVICE}-py3
DOCKER='sudo docker'

cp ../sources.list.ubuntu18.04 .

if [ ${DEVICE} == 'cpu' ] && [ ${TARGET} == 'deltann' ];then
  IMAGE=tensorflow/tensorflow:devel-py3
elif [ ${DEVICE} == 'gpu' ] && [ ${TARGET} == 'deltann' ];then
  IMAGE=tensorflow/tensorflow:devel-gpu-py3
elif [ ${DEVICE} == 'cpu' ] && [ ${TARGET} == 'delta' ];then
  IMAGE=tensorflow/tensorflow:latest-py3
else
  IMAGE=tensorflow/tensorflow:latest-gpu-py3
fi

$DOCKER pull $IMAGE

cat > dockerfile <<EOF
FROM ${IMAGE}
COPY sources.list.ubuntu18.04 /etc/apt/sources.list

# install tools 
COPY install.sh /install.sh
RUN /bin/bash /install.sh

#add user
ENV ROLE delta
RUN adduser --disabled-password --gecos '' --shell '/bin/bash' \$ROLE \
  && adduser \$ROLE sudo \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER \$ROLE 

# install gcc
COPY install_user.sh /ci/install_user.sh
RUN /bin/bash /ci/install_user.sh

CMD ["/bin/bash", "-c"]
EOF


# build image
$DOCKER build --no-cache=false -t delta:$TAG -f $MAIN_ROOT/docker/devel/dockerfile . || { echo "build ${TARGET} ${DEVICE} error"; exit 1; }

if [ -f dockerfile ] && [ ${SAVE_DOCKERFILE} == false ];then
  unlink dockerfile
fi


#push image
$DOCKER tag delta:$TAG zh794390558/delta:$TAG
$DOCKER push zh794390558/delta:$TAG

if [ $? == 0  ]
then
    echo "push successful";
    exit 0
else
    echo "push error";
    echo "make sure you have login docker"
    echo "plaese run 'docker login'"
    exit 10
fi
