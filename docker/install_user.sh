# fix for tf1.14 docker
# issue https://github.com/tensorflow/tensorflow/issues/29951
sudo apt-get update && sudo apt-get install -y --no-install-recommends gcc-4.8 g++-4.8 && \
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100 && \
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100 && \
  sudo update-alternatives --config gcc && \
  sudo update-alternatives --config g++ && \
  sudo apt-get clean && \
  sudo rm -rf /var/lib/apt/lists/*

