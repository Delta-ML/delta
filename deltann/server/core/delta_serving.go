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
	"delta/deltann/server/core/conf"
	"delta/deltann/server/core/handel"
	. "delta/deltann/server/core/model"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
	"os"
	"os/signal"
	"syscall"
)

const defaultPort = "8004"

// Options
type DeltaOptions struct {
	ServerPort     string
	ServerType     string
	DeltaModelYaml string
}

var deltaInterface DeltaInterface

func init() {
	listenSystemStatus()
}

func DeltaListen(opts DeltaOptions) error {
	conf.SetConfPath(opts.DeltaModelYaml)
	dParams := DeltaParam{opts.DeltaModelYaml}
	err := dParams.DeltaModelInit()
	if err != nil {
		return err
	}
	glog.Infof("start deltaModelRun...")
	router := gin.Default()

	relativePathRoot := "/v1/models/" + conf.DeltaConf.Model.Graph[0].Local.ModelType
	relativePathFull := relativePathRoot + "/versions/"
	relativePathFull = relativePathFull + conf.DeltaConf.Model.Graph[0].Version + ":" + opts.ServerType

	router.POST(relativePathFull, handel.DeltaPredictHandler)
	router.POST(relativePathRoot, handel.DeltaModelHandler)

	dPort := opts.ServerPort
	if dPort == "" {
		dPort = defaultPort
	}

	err = router.Run(":" + dPort)
	if err != nil {
		glog.Infof("delta serving init port  %s", dPort)
	}
	glog.Infof("delta serving DeltaPredictHandler port  %s  path %s", dPort, relativePathFull)
	glog.Infof("delta serving DeltaModelHandler port  %s  path %s", dPort, relativePathRoot)
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
