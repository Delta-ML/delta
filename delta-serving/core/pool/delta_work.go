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
package pool

import "C"
import (
	"delta/delta-serving/core/model"
	"github.com/golang/glog"
	"unsafe"
)

type DeltaJob struct {
	DeltaInputs interface{}
	Reply       string
	Done        chan DeltaJob
}

var DeltaJobQueue chan DeltaJob

func init() {
	DeltaJobQueue = make(chan DeltaJob, MaxQueue)
}

func (call DeltaJob) done() {
	select {
	case call.Done <- call:
	default:
	}
}

type DeltaWorker struct {
	DeltaHandel unsafe.Pointer
	WorkerPool  chan chan DeltaJob
	JobChannel  chan DeltaJob
	quit        chan bool
}

func NewWorker(workerPool chan chan DeltaJob, deltaHandel unsafe.Pointer) DeltaWorker {

	return DeltaWorker{
		DeltaHandel: deltaHandel,
		WorkerPool:  workerPool,
		JobChannel:  make(chan DeltaJob),
		quit:        make(chan bool)}
}

func (w DeltaWorker) Start() {
	go func() {
		for {
			w.WorkerPool <- w.JobChannel
			select {
			case job := <-w.JobChannel:
				//time.Sleep(1 * time.Millisecond)
				if w.DeltaHandel == nil {
					glog.Infof("w.DeltaHandel is nil")
					return
				}
				if modelResult, err := model.DeltaModelRun(job.DeltaInputs, w.DeltaHandel); err != nil {
					glog.Infof("DeltaModelRun error：%s", err.Error())
					job.Reply = err.Error()
				} else {
					glog.Infof("DeltaModelRun result %s", modelResult)
					job.Reply = modelResult
				}
				job.done()

			case <-w.quit:
				model.DeltaDestroyHandel(w.DeltaHandel)
				return
			}
		}
	}()
}

func (w DeltaWorker) Stop() {
	go func() {
		w.quit <- true
	}()
}
