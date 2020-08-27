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

import (
	. "delta/delta-serving/core/model"
	"github.com/golang/glog"
	"os"
)

type Dispatcher struct {
	WorkerPool chan chan DeltaJob
	maxWorkers int
}

var workerArray []DeltaWorker
var MaxWorker int
var MaxQueue int

func DeltaDispatcher(maxWorkers int, maxQueue int) *Dispatcher {
	MaxWorker = maxWorkers
	MaxQueue = maxQueue
	glog.Infof("MaxWorker %d  MaxQueue %d", MaxWorker, MaxQueue)
	pool := make(chan chan DeltaJob, maxWorkers)
	return &Dispatcher{WorkerPool: pool, maxWorkers: maxWorkers}
}

func (d *Dispatcher) StopWorkers() {
	for i := 0; i < d.maxWorkers; i++ {
		workerArray[i].Stop()
	}
	DeltaDestroyModel()
	close(DeltaJobQueue)
}

func (d *Dispatcher) Run() {
	for i := 0; i < d.maxWorkers; i++ {
		dHandel, err := DeltaCreateHandel()
		if err != nil {
			glog.Infof("DeltaCreateHandel error %s", err.Error())
			os.Exit(0)
			return
		}
		worker := NewWorker(d.WorkerPool, dHandel)
		worker.Start()
		workerArray = append(workerArray, worker)
	}

	go d.dispatch()
}

func (d *Dispatcher) dispatch() {
	for {
		select {
		case job := <-DeltaJobQueue:
			go func(job DeltaJob) {
				jobChannel := <-d.WorkerPool
				jobChannel <- job
			}(job)
		}
	}
}
