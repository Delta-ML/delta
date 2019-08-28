package main

import (
	."delta/deltann/server/core"
	"flag"
	"github.com/golang/glog"
)

	func main() {
		flag.Parse()
		defer glog.Flush()

		err := DeltaListen(DeltaOptions{":8004",""})
		if err != nil {
			glog.Fatalf("DeltaListen err %s",err.Error())
		}

}
