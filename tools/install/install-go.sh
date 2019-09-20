#! /bin/bash
set -euf -o pipefail

url="$(wget -qO- https://golang.org/dl/ | grep -oP 'https:\/\/dl\.google\.com\/go\/go([0-9\.]+)\.linux-amd64\.tar\.gz' | head -n 1 )"
latest="$(echo $url | grep -oP 'go[0-9\.]+' | grep -oP '[0-9\.]+' | head -c -2 )"
echo "Downloading latest Go for AMD64: ${latest}"
wget --quiet --continue --show-progress "${url}"
unset url

sudo tar -C /usr/local -xzf go"${latest}".linux-amd64.tar.gz
echo "Create the skeleton for your local users go directory"
mkdir -p ~/go/{bin,pkg,src}
echo "Setting up GOPATH"
echo "export GOPATH=~/go" >> ~/.profile && source ~/.profile
echo "Setting PATH to include golang binaries"
echo "export PATH='$PATH':/usr/local/go/bin:$GOPATH/bin" >> ~/.profile && source ~/.profile
echo "export GO111MODULE=on" >> ~/.profile && source ~/.profile

# Remove Download
rm go"${latest}".linux-amd64.tar.gz

# Print Go Version
/usr/local/go/bin/go version
