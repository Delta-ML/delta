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
	"delta/delta-serving/core/conf"
	"delta/delta-serving/core/handler"
	. "delta/delta-serving/core/model"
	. "delta/delta-serving/core/pool"
	"fmt"
	"github.com/gin-contrib/pprof"
	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
	"os"
	"os/signal"
	"syscall"
)

// Options
type DeltaOptions struct {
	Debug          bool
	ServerPort     string
	ServerType     string
	DeltaModelYaml string
}

var deltaInterface DeltaInterface
var defaultPort string
var dispatcher *Dispatcher

func init() {
	listenSystemStatus()
}

func DeltaListen(opts DeltaOptions) (*gin.Engine, error) {
	defer glog.Flush()
	conf.SetConfPath(opts.DeltaModelYaml)
	dParams := DeltaParam{DeltaYaml: opts.DeltaModelYaml}
	glog.Infof("start DeltaModelInit...")
	err := dParams.DeltaModelInit()
	glog.Infof("end DeltaModelInit...")
	if err != nil {
		return nil, err
	}
	glog.Infof("start DeltaDispatcher...")
	dispatcher = DeltaDispatcher(conf.DeltaConf.DeltaServingPoll.DeltaMaxWorker, conf.DeltaConf.DeltaServingPoll.DeltaMaxQueue)
	dispatcher.Run()

	glog.Infof("start deltaModelRun...")
	router := gin.Default()
	if opts.Debug {
		pprof.Register(router)
		gin.SetMode(gin.DebugMode)
	} else {
		gin.SetMode(gin.ReleaseMode)
	}
	relativePathRoot := "/v1/models/" + conf.DeltaConf.Model.Graph[0].Local.ModelType
	relativePathFull := relativePathRoot + "/versions/"
	relativePathFull = relativePathFull + conf.DeltaConf.Model.Graph[0].Version + ":" + opts.ServerType

	router.POST(relativePathFull, handler.DeltaPredictHandler)
	router.POST(relativePathRoot, handler.DeltaModelHandler)

	defaultPort = opts.ServerPort

	glog.Infof("delta serving DeltaPredictHandler path %s", relativePathFull)
	glog.Infof("delta serving DeltaModelHandler  path %s", relativePathRoot)
	return router, nil
}

func DeltaRun(router *gin.Engine) error {
	defer glog.Flush()
	err := router.Run(":" + defaultPort)
	if err != nil {
		glog.Infof("delta serving init port  %s", defaultPort)
		return err
	}
	return nil
}

func listenSystemStatus() {
	c := make(chan os.Signal)
	signal.Notify(c, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)
	go func() {
		for s := range c {
			switch s {
			case syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT:
				dispatcher.StopWorkers()
			default:
				fmt.Println("other:", s)
			}
		}
	}()
}
