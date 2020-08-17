# Deltann Serving

## Table of Contents
- [Quick Start](#quick-start)
- [RESTful API](#restful-api)
- [Model status API](#model-status-api)
- [Predict API](#predict-api)
- [Request format](#Request-format)
- [Demo code](#demo-code)

## Quick start

After `delta/dpl/run.sh` is successfully executed, run the following command again.

```shell
$ cd example
$ ./build.sh
```

```shell
$ cd output/delta-service
$ ./run.sh start &
$ cat log/delta-service.INFO
...
I0927 07:26:37.897093    1057 delta_serving.go:76] delta serving DeltaPredictHandler path /v1/models/saved_model/versions/1:predict
I0927 07:26:37.897394    1057 delta_serving.go:77] delta serving DeltaModelHandler  path /v1/models/saved_model/versions/1
[GIN-debug] Listening and serving HTTP on :8004
```

 ## RESTful API
 
 API interface reference [TensorFlow](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md)
 
 In case of error, all APIs will return a JSON object in the response body with error as key and the error message as the value:
 
 ```sh
 {
   "error": <error message string>
 }
 ```

### Model status API
It returns the status of a model in the ModelServer.
```sh
POST http://host:port/v1/models/${MODEL_NAME}

{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

#### Predict API
```sh
POST http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]:predict
``` 
 
 #### Request format
 
The request body for predict API must be JSON object formatted as follows:

```javascript
{
  "inputs": <value>
}
```

 
 ### Demo code
  ```sh   
 $cd delta-serving/examples/
 $ cat main.go
 ```

 ```go
 package main
 import (
     . "delta/delta-serving/core"
     "flag"
     "github.com/golang/glog"
     "os"
 )
 
 func main() {
     deltaPort := flag.String("port", "none", "set http listen port")
     deltaYaml := flag.String("yaml", "none", "set delta model yaml conf")
     deltaType := flag.String("type", "none", "set server type：predict | classify")
     deltaDebug := flag.Bool("debug", false, "set debug environment：true | false")
     flag.Parse()
     defer glog.Flush()
     var deltaOptions = DeltaOptions{
         Debug:          *deltaDebug,
         ServerPort:     *deltaPort,
         ServerType:     *deltaType,
         DeltaModelYaml: *deltaYaml,
     }
     r, err := DeltaListen(deltaOptions)
     if err != nil {
         glog.Fatalf("DeltaListen err %s", err.Error())
         os.Exit(1)
     }
     err = DeltaRun(r)
     if err != nil {
         glog.Fatalf("DeltaRun err %s", err.Error())
         os.Exit(1)
     }
 }
 ```
