FROM tensorflow/tensorflow:devel-gpu
COPY sources.list.ubuntu18.04 /etc/apt/sources.list

# install tools 
COPY install.sh /install.sh
RUN /bin/bash /install.sh

CMD ["/bin/bash", "-c"]
