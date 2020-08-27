/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package model

/*
#cgo CFLAGS: -I${SRCDIR}/../../dpl/output/include
#cgo LDFLAGS: -L${SRCDIR}/../../dpl/output/lib/deltann  -ldeltann  -L${SRCDIR}/../../dpl/output/lib/tensorflow -ltensorflow_cc -ltensorflow_framework -L${SRCDIR}/../../dpl/output/lib/custom_ops -lx_ops  -lm -fPIC -O2  -lstdc++  -lz -lpthread
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <c_api.h>
*/
import "C"
import (
	"bytes"
	"delta/delta-serving/core/conf"
	"delta/delta-serving/core/types"
	"encoding/gob"
	"encoding/json"
	"errors"
	"github.com/golang/glog"
	"unsafe"
)

var model C.ModelHandel

type DeltaInterface interface {
	DeltaModelInit() error
	DeltaModelRun() error
	DeltaDestroy()
}

type DeltaParam struct {
	DeltaYaml string
}

func (dParam DeltaParam) DeltaModelInit() error {
	yamlFile := C.CString(dParam.DeltaYaml)
	defer C.free(unsafe.Pointer(yamlFile))
	model = C.DeltaLoadModel(yamlFile)
	if model == nil {
		return errors.New("deltaLoadModel failed")
	}

	return nil
}

func DeltaCreateHandel() (unsafe.Pointer, error) {
	deltaInf := C.DeltaCreate(model)
	if deltaInf == nil {
		return nil, errors.New("deltaCreate failed")
	}
	return unsafe.Pointer(deltaInf), nil
}

func DeltaModelRun(valueInputs interface{}, cInf unsafe.Pointer) (string, error) {
	inf := *(*C.InferHandel)(unsafe.Pointer(&cInf))

	if inf == nil {
		return "", errors.New("C.InferHandel nil")
	}

	if conf.DeltaConf.DeltaServingPoll.DeltaApiType == types.DJson {
		valPtr := C.CString(valueInputs.(string))
		defer C.free(unsafe.Pointer(valPtr))
		C.DeltaSetJsonInputs(inf, valPtr)
	}else if conf.DeltaConf.DeltaServingPoll.DeltaApiType == types.DString {
		inNum := C.int(len(conf.DeltaConf.Model.Graph[0].Inputs))
		var ins C.Input
		deltaPtr := C.CString(valueInputs.(string))
		defer C.free(unsafe.Pointer(deltaPtr))
		ins.ptr = unsafe.Pointer(deltaPtr)
		// len(valueInputs.(string)) + 1   for text /0
		ins.nelms = C.int(len(valueInputs.(string)) + 1)

		inputName := C.CString(conf.DeltaConf.Model.Graph[0].Inputs[0].Name)
		defer C.free(unsafe.Pointer(inputName))
		ins.input_name = inputName

		graphName := C.CString(conf.DeltaConf.Model.Graph[0].Name)
		defer C.free(unsafe.Pointer(graphName))
		ins.graph_name = graphName
		glog.Infof("before DeltaSetInputs")
		C.DeltaSetInputs(inf, &ins, inNum)
	}else{
		return "", errors.New("unsupported data type")
	}

	C.DeltaRun(inf)
	outNum := C.DeltaGetOutputCount(inf)
	glog.Infof("The output num is %d", outNum)
	var dynaArr []C.float
	var data *C.float
	for i := 0; i < int(outNum); i++ {

		byteSize := C.DeltaGetOutputByteSize(inf, C.int(i))
		data = (*C.float)(C.malloc(C.size_t(byteSize)))
		C.DeltaCopyToBuffer(inf, C.int(i), unsafe.Pointer(data), byteSize)
		num := byteSize / C.sizeof_float

		for j := 0; j < int(num); j++ {
			p := (*[1 << 30]C.float)(unsafe.Pointer(data))
			dynaArr = append(dynaArr, p[j])
		}
	}
	defer C.free(unsafe.Pointer(data))
	pagesJson, err := json.Marshal(dynaArr)
	if err != nil {
		glog.Infof("Cannot encode to JSON %s ", err.Error())
		return "", err
	}
	glog.Infof("Success encode to JSON %s ", pagesJson)
	return string(pagesJson), nil
}

func DeltaDestroyHandel(cInf unsafe.Pointer) {
	inf := *(*C.InferHandel)(unsafe.Pointer(&cInf))
	C.DeltaDestroy(inf)
}

func DeltaDestroyModel() {
	C.DeltaUnLoadModel(model)
}

func GetBytes(key interface{}) ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(key)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
