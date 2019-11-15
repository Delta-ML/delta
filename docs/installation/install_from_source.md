# Install from the source code

To install from the source code, We use [conda](https://conda.io/) to
install required packages. Please
[install conda](https://conda.io/en/latest/miniconda.html) if you do not
have it in your system.

Also, we provide two options to install DELTA, `nlp` version or `full`
version. `nlp` version needs minimal requirements and only installs NLP
related packages:

```shell
# Run the installation script for NLP version, with CPU or GPU.
cd tools
./install/install-delta.sh nlp [cpu|gpu]
```

**Note**: Users from mainland China may need to set up conda mirror sources, see [./tools/install/install-delta.sh](tools/install/install-delta.sh) for details.

If you want to use both NLP and speech packages, you can install the `full` version. The full version needs [Kaldi](https://github.com/kaldi-asr/kaldi) library, which can be pre-installed or installed using our installation script.

```shell
cd tools
# If you have installed Kaldi
KALDI=/your/path/to/Kaldi ./install/install-delta.sh full [cpu|gpu]
# If you have not installed Kaldi, use the following command
# ./install/install-delta.sh full [cpu|gpu]
```

To verify the installation, run:

```shell
# Activate conda environment
conda activate delta-py3.6-tf2.0.0
# Or use the following command if your conda version is < 4.6
# source activate delta-py3.6-tf2.0.0

# Add DELTA enviornment
source env.sh

# Generate mock data for text classification.
pushd egs/mock_text_cls_data/text_cls/v1
./run.sh
popd

# Train the model
python3 delta/main.py --cmd train_and_eval --config egs/mock_text_cls_data/text_cls/v1/config/han-cls.yml
```
