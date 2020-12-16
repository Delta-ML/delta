##MKL #export PATH=/opt/intel/mkl/bin:$PATH
##export TF_MKL_ROOT=/nfs/project/intel/mkl
#
#export TF_MKL_ROOT=/opt/intel/mkl
#export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/opt/intel/lib:$LD_LIBRARY_PATH
#
#
#export LD_LIBRARY_PATH=/nfs/project/intel/mkl-dnn/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/nfs/project/mkl-dnn/external/mklml_lnx_2019.0.3.20190125:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/nfs/project/mkl-dnn/external/mklml_lnx_2019.0.3.20190125/lib:$LD_LIBRARY_PATH
#
#export LIBRARY_PATH=/nfs/project/mkl-dnn/external/mklml_lnx_2019.0.3.20190125/:$LIBRARY_PATH
#export LIBRARY_PATH=/nfs/project/mkl-dnn/external/mklml_lnx_2019.0.3.20190125/lib:$LIBRARY_PATH
#
#export TF_NEED_MKL="1"
#
#
#export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH
#PULLER_TIMEOUT=10000
#bazel build \
#    --config=mkl \
#    --http_timeout_scaling=10.0 \
#    --loading_phase_threads=1 \
#    --verbose_failures \
#    -c opt --cxxopt=-std=c++11 \
#    --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 \
#    --config=v2 \
#    //tensorflow:libtensorflow_cc.so \
#    //tensorflow/tools/benchmark:benchmark_model \
#    --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
#    --action_env="LIBRARY_PATH=${LIBRARY_PATH}"
#
#
##gpu
#PYTHON_LIBRARY_PATH=/tmp-data/zhanghui/envs/venv/lib/python3.6/site-packages
# --output_base=/nfs/private/tf build --verbose_failures -s 


##bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma  tensorflow/tools/benchmark:benchmark_model


# build gpu
bazel \
    --output_base /nfs/private/tf\
    build \
    --config=cuda \
    --config=nonccl \
    --config=noaws \
    --verbose_failures \
    //tensorflow:libtensorflow_cc.so \
    //tensorflow/tools/benchmark:benchmark_model \
    //tensorflow/tools/pip_package:build_pip_package
