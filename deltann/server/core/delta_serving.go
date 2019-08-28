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
package core

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo LDFLAGS: -L${SRCDIR}/lib  -lm  -lstdc++  -lz -lpthread
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
*/
import "C"

import (
	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
)

const defaultPort = ":8004"

// Options allows configuring the started agent.
type DeltaOptions struct {
	ServerPort         string
	ServerRelativePath string
}

func DeltaListen(opts DeltaOptions) error {
	router := gin.Default()
	router.POST("/delta/api/:name/*action", func(context *gin.Context) {
		glog.Info("hello")
	})

	dPort := opts.ServerPort
	if dPort == "" {
		dPort = defaultPort
	}

	err := router.Run(dPort)
	if err != nil {
		glog.Infof("delta serving init port  %s", dPort)
	}

	return err
}
