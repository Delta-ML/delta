# DELTA-NN architecture 

1. features
1. compile
1. package
  * TF-Serving
  * Serving
  * Embedding
  * Client

## Features

* tiny size
* pack all by docker
* supporting custom-op
* exporting only C api
* compatibility TF-Serving and Serving RESTful API 
* compatibility with Could and Edge usage
* supporting multi graphs inference
* supporting multi modal application, e.g KWS(Edge) + ASR(Could)

## Compile
How to compile `delta-nn`.


## TF-Serving
### Suport Custom-Op
Compile TF-Serving with custom Ops.

### Serving with Docker
* [Tensorflow Serving with Docker](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md)
* [Serving with Docker using your GPU](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md#serving-with-docker-using-your-gpu)
* [Tensorflow Serving with Kubernetes](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_kubernetes.md)



## Serving
Using `go` to wraper DELTA-NN supporting HTTP/HTTPS protocal.
### Support Engines
* TF
* TFTRT

### REST API
Compatibility with [TF-Serving](https://github.com/tensorflow/serving) 
[RESTful API](https://www.tensorflow.org/tfx/serving/api_rest).

* input tensors in [row format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_row_format)
* input tensors in [column format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_column_format)

### Data Exchange 
Using [json](https://github.com/nlohmann/json) to exchange data with Client and Serving,
since it support  [CBOR](https://cbor.io/) and [BSON](http://bsonspec.org/) specification.



## Embedding
### Support Engines
* TFLite

### Core
DELTA-NN using [TFLite](https://www.tensorflow.org/lite/guide) as backend engine.



## Client
### Support Engines
* TF
* TFTRT
* TFLite

### HTTP/HTTPS Client
DELTA-NN using [mbedtls](https://github.com/ARMmbed/mbedtls) as SSL/TLS library.

It's an OpenSSL alternative libraray, which has many features:
* Fully featured SSL/TLS and cryptography library
* Easy integration with a small memory footprint
* Easy to understand and use with a clean API
* Easy to reduce and expand the code
* Easy to build with no external dependencies
* Extremely portable

