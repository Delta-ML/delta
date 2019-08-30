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
#cgo CFLAGS: -I${SRCDIR}/../dpl/output/include
#cgo LDFLAGS: -L${SRCDIR}/../dpl/output/lib/deltann  -ldeltann  -L${SRCDIR}/../dpl/output/lib/tensorflow -ltensorflow_cc -ltensorflow_framework -L${SRCDIR}/../dpl/output/lib/custom_ops -lx_ops  -lm -fPIC -O2  -lstdc++  -lz -lpthread
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <c_api.h>
*/
import "C"
import (
	"github.com/golang/glog"
	"unsafe"
)

var inf C.InferHandel
var model C.ModelHandel

func DeltaModelInit(yaml string) error {
	yamlFile := C.CString(yaml)
	defer C.free(unsafe.Pointer(yamlFile))
	model = C.DeltaLoadModel(yamlFile)
	inf = C.DeltaCreate(model)
	return nil
}

func DeltaModelRun(uText string) error {
	inNum := C.int(1)
	var ins C.Input

	text := C.CString(uText)
	defer C.free(unsafe.Pointer(text))
	ins.ptr = unsafe.Pointer(text)

	ins.size = 1

	inputName := C.CString("input_sentence")
	defer C.free(unsafe.Pointer(inputName))
	ins.input_name = inputName

	graphName := C.CString("default")
	defer C.free(unsafe.Pointer(graphName))
	ins.graph_name = graphName

	glog.Infof("ins %s", ins)

	C.DeltaSetInputs(inf, &ins, inNum)
	C.DeltaRun(inf)
	outNum := C.DeltaGetOutputCount(inf)
	glog.Infof("The output num is %d", outNum)

	for i := 0; i < int(outNum); i++ {

		byteSize := C.DeltaGetOutputByteSize(inf, C.int(i))
		data := (*C.float)(C.malloc(C.size_t(byteSize)))
		C.DeltaCopyToBuffer(inf, C.int(i), unsafe.Pointer(data), byteSize)
		num := byteSize / C.sizeof_float
		for j := 0; j < int(num); j++ {
			p := (*[1 << 30]C.float)(unsafe.Pointer(data))
			glog.Infof("score is %f", p[j])
		}
		C.free(unsafe.Pointer(data))
	}

	return nil
}

func DeltaDestroy() error {
	C.DeltaDestroy(inf)
	C.DeltaUnLoadModel(model)
	return nil
}
