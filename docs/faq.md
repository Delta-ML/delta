# FAQ

## Install

> 1. How to speed up the installation?

If you are a user from mainland China, you can use the comments code in 
`tools/install/install-delta.sh`.

```shell
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

>  2. `CondaValueError: prefix already exists:
>  ../miniconda3/envs/delta-py3.6-tf2.0.0` or `ERROR: unknown command "config"`

Please update your conda by:

```shell
conda update -n base -c defaults conda
```

> 3. Custom operator error:
> `tensorflow.python.framework.errors_impl.NotFoundError:
> /../delta/delta/layers/ops/x_ops.so: undefined symbol:
> _ZN10tensorflow8str_util9LowercaseEN4absl11string_viewE`

This error always raise when you use the tensorflow installed by conda instead of pip. Conda use more high level gcc than pip dose to compile tensorflow. In this case, compilation of custom op with g++ 4.8 may cause this error.

You can use `conda install -c conda-forge cxx-compiler` to update the g++ version under your conda env.

then, compile custom op again：

```shell
pushd delta/layers/ops/
./build.sh delta
popd
```

> 4. Segmentation fault.
> 0x00007fff48e930d4 in tensorflow::shape_inference::UnchangedShape(tensorflow::shape_inference::InferenceContext*) ()

This error always raise when you use the tensorflow installed by pip instead of conda. The pip is compiled by g++ 4.8. In this case, you need to install g++ 4.8 on your system and re-compile your custom op again.

The error no.3 and no.4 are similar questions. The principle is to keep the g++ version for tensorflow compilation and custom compilation same. You need to upgrade or downgrade your g++ according to the cases.
