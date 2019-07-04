#!/bin/bash

graph=$1

ROOT=${MAIN_ROOT}/tools/tensorflow

${ROOT}/bazel-bin/tensorflow/python/tools/print_selective_registration_header \
    --graphs=$graph > ops_to_register.h
