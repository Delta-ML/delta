FROM tensorflow/tensorflow:2.3.0
COPY sources.list.ubuntu18.04 /etc/apt/sources.list

# install tools 
COPY install.sh /install.sh
RUN /bin/bash /install.sh

COPY requirements.txt /ci/requirements.txt
WORKDIR /ci
RUN sudo pip --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip && pip --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple --user -r requirements.txt

CMD ["/bin/bash", "-c"]
