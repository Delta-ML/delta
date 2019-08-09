# FAQ

## Install

> How to speed up the installation?

If you are a user from mainland China, you can use the comments code in 
`tools/install/install-delta.sh`.

```shell
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

>  `CondaValueError: prefix already exists:
>  ../miniconda3/envs/delta-py3.6-tf1.14` or `ERROR: unknown command "config"`

Please update your conda by:

```shell
conda update -n base -c defaults conda
```

> Custom operator error:
> `tensorflow.python.framework.errors_impl.NotFoundError:
> /../delta/delta/layers/ops/x_ops.so: undefined symbol:
> _ZN10tensorflow8str_util9LowercaseEN4absl11string_viewE`

Currently, compilation of custom op with g++ 4.8 may cause this error.

You can use `conda install -c conda-forge cxx-compiler` to update the
g++ version under your conda env.

then, compile custom op again：

```shell
pushd delta/layers/ops/
./build.sh delta
popd
```
