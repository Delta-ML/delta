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
#cgo LDFLAGS: -L${SRCDIR}/../dpl/output/lib/deltann  -ldeltann -L${SRCDIR}/../dpl/output/lib/tensorflow -ltensorflow_cc -ltensorflow_framework -L${SRCDIR}/../dpl/output/lib/custom_ops -lx_ops  -lm  -lstdc++  -lz -lpthread
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <c_api.h>
extern int DeltaGoInit(char* yamDeltaGoInitl_file);
extern int DeltaGoRun();
extern int DeltaGoDestroy();
*/
import "C"

func DeltaModelInit(yaml string) error {
	yamlFile := C.CString(yaml)
	C.DeltaGoInit(yamlFile)
	return nil
}

func DeltaModelRun() error {
	C.DeltaGoRun()
	return nil
}

func DeltaDestroy() error {
	C.DeltaGoDestroy()
	return nil
}
