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
package handler

import (
	. "delta/delta-serving/core/pool"
	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
	"net/http"
)

// Binding from JSON
type DeltaRequest struct {
	DeltaSignatureName string        `form:"signature_name" json:"signature_name" `
	DeltaInstances     []interface{} `form:"instances" json:"instances" `
	DeltaInputs        interface{}   `form:"inputs" json:"inputs" `
}

func DeltaPredictHandler(context *gin.Context) {
	var json DeltaRequest
	if err := context.ShouldBindJSON(&json); err != nil {
		context.JSON(http.StatusBadRequest, gin.H{"error": "DeltaRequest information is not complete"})
		return
	}

	var reply string
	call := DeltaPackJob(json, reply, make(chan DeltaJob, 1))
	result := <-call.Done

	context.JSON(http.StatusOK, gin.H{"predictions": result.Reply})

}

func DeltaPackJob(req DeltaRequest, reply string, done chan DeltaJob) DeltaJob {
	defer glog.Flush()
	work := DeltaJob{DeltaInputs: req.DeltaInputs, Reply: reply, Done: done}
	DeltaJobQueue <- work
	return work
}
