#!/bin/bash 

if [ $# != 3 ];then
  echo "usage: $0 [ci|delta|deltann] [cpu|gpu] [push|dockerfile]"
  exit 1
fi

if [ -z $MAIN_ROOT ];then
  pushd .. && . env.sh && popd
fi

# generate dockerfile
TARGET=$1 #delta or deltann
DEVICE=$2 # cpu or gpu
MODE=$3 # push image or generate dockerfile

SAVE_DOCKERFILE=false
if [ $MODE == 'dockerfile' ];then
  SAVE_DOCKERFILE=true
fi

DOCKERFILE=$MAIN_ROOT/docker/dockerfile.${TARGET}.${DEVICE} 
if [ -f $DOCKERFILE ];then
  unlink $DOCKERFILE
fi


TAG=1.14.0-${TARGET}-${DEVICE}-py3
DOCKER='sudo docker'
PIP_INSTALL="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/"

on_exit() {
  rm requirements.txt
  if [ $SAVE_DOCKERFILE == false ];then
    rm $DOCKERFILE 
  fi
}
trap on_exit HUP INT PIPE QUIT TERM EXIT


if [ ${DEVICE} == 'cpu' ] && [ ${TARGET} == 'deltann' ];then
  IMAGE=tensorflow/tensorflow:devel-py3
elif [ ${DEVICE} == 'gpu' ] && [ ${TARGET} == 'deltann' ];then
  IMAGE=tensorflow/tensorflow:devel-gpu-py3
elif [ ${DEVICE} == 'cpu' ] && [ ${TARGET} == 'delta' ] || [ ${TARGET} == 'ci'];then
  IMAGE=tensorflow/tensorflow:1.14.0-py3
elif [ ${DEVICE} == 'gpu' ] && [ ${TARGET} == 'delta' ] || [ ${TARGET} == 'ci' ];then
  IMAGE=tensorflow/tensorflow:1.14.0-gpu-py3
else
  echo "no support target or device"
  exit -1
fi

# generate dockerfile
cp ../sources.list.ubuntu18.04 .
cp $MAIN_ROOT/tools/requirements.txt .


cat > $DOCKERFILE <<EOF
FROM ${IMAGE}
COPY sources.list.ubuntu18.04 /etc/apt/sources.list

# install tools 
COPY install.sh /install.sh
RUN /bin/bash /install.sh

EOF

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

cat >> $DOCKERFILE <<EOF
# install gcc
COPY install_user.sh /ci/install_user.sh
RUN /bin/bash /ci/install_user.sh

EOF

if [ ${TARGET} == 'ci' ];then
# install python pakcages
cat >> $DOCKERFILE <<EOF
COPY requirements.txt /ci/requirements.txt
WORKDIR /ci
RUN $PIP_INSTALL --upgrade pip && $PIP_INSTALL --user -r requirements.txt

EOF
fi

cat >> $DOCKERFILE <<EOF
CMD ["/bin/bash", "-c"]
EOF


if [ $SAVE_DOCKERFILE == false ];then
  # pull latest image
  $DOCKER pull $IMAGE
  
  # build image
  $DOCKER build --no-cache=false -t delta:$TAG -f $DOCKERFILE . || { echo "build ${TARGET} ${DEVICE} error"; exit 1; }
  
  #push image
  $DOCKER tag delta:${TAG} zh794390558/delta:${TAG}
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

if [ -f $DOCKERFILE ] && [ ${SAVE_DOCKERFILE} == false ];then
  unlink $DOCKERFILE 
fi



