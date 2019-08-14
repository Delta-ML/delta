# Install on macOS

Running DELTA training on a macOS is mostly the same as running on Linux, except some minor differences.

## Python environment

You need to set up a working Python 3.6.x environment, either by using conda or manually build from source.
You can follow the instructions in `manual_setup.md` to set up python and the required packages, e.g. Tensorflow.
Note: `tensorflow-gpu` requires nvidia GPU, which might not be supported the latest macOS versions. You may want to use the `tensorflow` package (no -gpu postfix) instead. Some models that uses cuDNN implementations will not work without a CUDA GPU however.

## Other requirements

### Notes for Kaldi

Building and running Kaldi on a macOS requires `wget`, `gawk` and other utilities which need to be installed via `Homebrew`. See `https://brew.sh` for details.
```shell
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install wget gawk grep
```
Also the `mmseg` package for Python2 is needed:
```shell
pip2 install mmseg
```

Then follow `manual_setup.md` / `DELTA install` section to install 3rd-party dependencies.

