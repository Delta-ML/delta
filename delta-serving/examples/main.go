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
package main

import (
	. "delta/delta-serving/core"
	"flag"
	"github.com/golang/glog"
	"os"
)

func main() {
	deltaPort := flag.String("port", "none", "set http listen port")
	deltaYaml := flag.String("yaml", "none", "set delta model yaml conf")
	deltaType := flag.String("type", "none", "set server type：predict | classify")
	deltaDebug := flag.Bool("debug", false, "set debug environment：true | false")
	flag.Parse()
	defer glog.Flush()
	var deltaOptions = DeltaOptions{
		Debug:          *deltaDebug,
		ServerPort:     *deltaPort,
		ServerType:     *deltaType,
		DeltaModelYaml: *deltaYaml,
	}

	r, err := DeltaListen(deltaOptions)
	if err != nil {
		glog.Fatalf("DeltaListen err %s", err.Error())
		os.Exit(1)
	}
	err = DeltaRun(r)
	if err != nil {
		glog.Fatalf("DeltaRun err %s", err.Error())
		os.Exit(1)
	}
}
