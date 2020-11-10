FROM tensorflow/tensorflow:2.3.0
COPY sources.list.ubuntu18.04 /etc/apt/sources.list

# install tools 
COPY install.sh /install.sh
RUN /bin/bash /install.sh

RUN sudo mkdir workspace
RUN cd /workspace && git clone --depth 1 https://github.com/didi/delta.git
RUN cd /workspace/delta/tools && make basic
WORKDIR /workspace/delta

CMD ["/bin/bash", "-c"]
