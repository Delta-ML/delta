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
	ServerPort string
	ServerRelativePath string
}

func DeltaListen(opts DeltaOptions) error{
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
		glog.Infof("delta serving init port  %s",dPort)
	}
	return err
}





