# dpl

Main root is `run.sh`, model input is `model` dir, output is `output` dir.

## input

Putting `saved_model` and `model.yaml` into `model` dir and then run `run.sh` to convert model, build library.

```text
model/
├── model.yaml
└── saved_model
    ├── saved_model.pbtxt
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index

2 directories, 4 files
```

## output

All things need to deploy model are in `output` dir.

```text
output/
├── include
│   └── c_api.h
├── lib
│   ├── custom_ops
│   │   └── libx_ops.so
│   ├── deltann
│   │   ├── libdeltann.a
│   │   └── libdeltann.so
│   ├── tensorflow
│   │   ├── libtensorflow_cc.so -> libtensorflow_cc.so.2
│   │   ├── libtensorflow_cc.so.2 -> libtensorflow_cc.so.2.0.0
│   │   ├── libtensorflow_cc.so.2.0.0
│   │   ├── libtensorflow_cc.so.2.0.0-2.params
│   │   ├── libtensorflow_framework.so -> libtensorflow_framework.so.2
│   │   ├── libtensorflow_framework.so.2 -> libtensorflow_framework.so.2.0.0
│   │   ├── libtensorflow_framework.so.2.0.0
│   │   └── libtensorflow_framework.so.2.0.0-2.params
│   └── tflite
└── model
    └── saved_model
        └── 1
            ├── model.yaml
            ├── saved_model.pbtxt
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index

10 directories, 16 files
```

