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

import (
	. "delta/deltann/server/model"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
)

const defaultPort = ":8004"

// Options
type DeltaOptions struct {
	ServerPort         string
	ServerRelativePath string
	DeltaModelYaml     string
}

// Binding from JSON
type DeltaRequest struct {
	DeltaType             string `form:"delta_type" json:"delta_type" binding:"required"`
	DeltaRawText          string `form:"delta_raw_text" json:"delta_raw_text"`
	DeltaModelInputSize   int    `form:"delta_model_input_size" json:"delta_model_input_size"  binding:"required"`
	DeltaModelInputName   string `form:"delta_model_input_name" json:"delta_model_input_name"  binding:"required"`
	DeltaModelInputNumber int    `form:"delta_model_input_number" json:"delta_model_input_number"  binding:"required"`
	DeltaModelGraphName   string `form:"delta_model_graph_name" json:"delta_model_graph_name"  binding:"required"`
}

const (
	DeltaNlp = "nlp"
	DeltaAsr = "asr"
)

var deltaInterface DeltaInterface

func init() {
	listenSystemStatus()
}

func DeltaListen(opts DeltaOptions) error {
	dParams := DeltaParam{opts.DeltaModelYaml}
	err := dParams.DeltaModelInit()
	if err != nil {
		return err
	}
	glog.Infof("start deltaModelRun...")
	router := gin.Default()
	router.POST(opts.ServerRelativePath, func(context *gin.Context) {

		var json DeltaRequest
		if err := context.ShouldBindJSON(&json); err != nil {
			context.JSON(http.StatusBadRequest, gin.H{"error": "DeltaRequest information is not complete"})
			return
		}

		switch json.DeltaType {
		case DeltaNlp:
			modelResult, err := DeltaModelRun(json.DeltaRawText)
			if err != nil {
				context.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}
			context.JSON(http.StatusOK, gin.H{"outputs": modelResult})
		case DeltaAsr:
			context.JSON(http.StatusBadRequest, gin.H{"error": "Coming soon"})
		default:
			context.JSON(http.StatusBadRequest, gin.H{"error": "Delta does not support current data types"})
		}

	})

	dPort := opts.ServerPort
	if dPort == "" {
		dPort = defaultPort
	}

	err = router.Run(dPort)
	if err != nil {
		glog.Infof("delta serving init port  %s", dPort)
	}

	return nil
}

func listenSystemStatus() {
	c := make(chan os.Signal)
	signal.Notify(c, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT, syscall.SIGUSR1, syscall.SIGUSR2)
	go func() {
		for s := range c {
			switch s {
			case syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM:
				fmt.Println("exit:", s)
				DeltaDestroy()
			case syscall.SIGUSR1:
				fmt.Println("usr1", s)
			case syscall.SIGUSR2:
				fmt.Println("usr2", s)
			default:
				fmt.Println("other:", s)
			}
		}
	}()
}
