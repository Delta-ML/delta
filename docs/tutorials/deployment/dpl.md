# Deployment scripts - dpl
The `dpl` directory is for orginating model `config`, `convert`, `testing`, `benchmarking` and `serving`.

## Inputs & Outputs

After model is exported as `SavedModel`, we recommend using [Netron](https://github.com/lutzroeder/netron) 
to view the neural network model, then getting the `inputs` and `outputs` names. 


Other tools to determine the inputs/outputs for GraphsDef protocol buffer:
* [summarize_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs)

* TensorBoard
To visualize a .pb file, use the [import_pb_to_tensorboard.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py) script like below:

```bash
python import_pb_to_tensorboard.py --model_dir <model path> --log_dir <log dir path>
```

* TFLite
Run the [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) script with bazel:
```bash
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```


## Model directory

Putting the `SavedModel` under `dpl/model` directory, config the `dpl/model/model.yaml` as it is.

## Graph Convert
        
Running `dpl/gadpter/run.sh` to convert model to other model format,
e.g. `tflite`, `tftrt`, `ngraph`, `onnix`, `coreml` and so on.

## Build Packages

All packges build under `docker` env, see `docker/dpl`.

* build tensorflow cpu
* build tensorflow gpu
* build tensorflow with TensorRT
* build tensorflow lite cpu
* build tensorflow lite Android 
* build tensorflow lite IOS
* build DELTA-NN with dependent packages
* build unit-test
* build examples under DELTA-NN

## Testing

Do belows testing under `docker` env, if all passed, then deployment the model:

* unit testing
* integration testing
* smoke testing
* stress testing

## AB Testing
If model is better than old model by `metrics` and `RTF`, then we push it to Could or Edge.


## Deployment

### Deploy Mode

For Could, deployment as belows mode:

1. DELTA-NN Serving
  * DELTA-NN TF CPU
  * DELTA-NN TFTRT GPU
  * DELTA-NN Client 

1. DELTA-NN TF-Serving

For Edge, as:
* DELTA-NN TFLite
* DELTA-NN Client 

### Deploy Env
For Could, pack library, bin and model into `docker`, then using `K8s+docker` to depoyment.
For Edge, pack library, bin and model as `tarball`.

