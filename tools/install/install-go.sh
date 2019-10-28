#! /bin/bash

#url="$(wget -qO- https://golang.org/dl/ | grep -oP 'https:\/\/dl\.google\.com\/go\/go([0-9\.]+)\.linux-amd64\.tar\.gz' | head -n 1 )"
#latest="$(echo $url | grep -oP 'go[0-9\.]+' | grep -oP '[0-9\.]+' | head -c -2 )"
#echo "Downloading latest Go for AMD64: ${latest}"
latest="1.13.3"
url="https://mirrors.ustc.edu.cn/golang/go${latest}.linux-amd64.tar.gz"
wget --quiet --continue --show-progress "${url}"
unset url

sudo tar -C /usr/local -xzf go"${latest}".linux-amd64.tar.gz

echo "Create the skeleton for your local users go directory"
mkdir -p ~/go/{bin,pkg,src}

echo "Setting up GOPATH"
echo "export GOPATH=~/go" >> ../go.env

echo "Setting PATH to include golang binaries"
echo "export PATH='$PATH':/usr/local/go/bin:$GOPATH/bin" >> ../go.env
echo "export GO111MODULE=on" >> ../go.env

# Remove Download
rm go"${latest}".linux-amd64.tar.gz

# Print Go Version
/usr/local/go/bin/go version

pushd ../
source go.env
popd
