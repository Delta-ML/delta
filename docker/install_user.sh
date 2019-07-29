# fix for tf1.14 docker
# issue https://github.com/tensorflow/tensorflow/issues/29951
sudo apt-get install gcc-4.8 g++-4.8 -y && \
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100 && \
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100 && \
  sudo update-alternatives --config gcc && \
  sudo update-alternatives --config g++

