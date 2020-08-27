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
package conf

import (
	"github.com/golang/glog"
	"gopkg.in/yaml.v2"
	"io/ioutil"
)

type DeltaConfig struct {
	Model            DeltaModel       `yaml:"model"`
	RunTime          DeltaRunTime     `yaml:"runtime"`
	DeltaServingPoll DeltaServingConf `yaml:"serving"`
}

type DeltaServingConf struct {
	DeltaApiType string `yaml:"api_type"`
	DeltaMaxWorker int `yaml:"max_worker"`
	DeltaMaxQueue  int `yaml:"max_queue"`
}

type DeltaModel struct {
	CustomOpsPath string        `yaml:"custom_ops_path"`
	Graph         []DeltaGraphs `yaml:"graphs"`
}

type DeltaGraphs struct {
	Id      int           `yaml:"id"`
	Name    string        `yaml:"name"`
	Engine  string        `yaml:"engine"`
	Version string        `yaml:"version"`
	Local   DeltaLocal    `yaml:"local"`
	Remote  DeltaRemote   `yaml:"remote"`
	Inputs  []DeltaInputs `yaml:"inputs"`
	Outputs []DeltaInputs `yaml:"outputs"`
}

type DeltaLocal struct {
	Path      string `yaml:"path"`
	ModelType string `yaml:"model_type"`
}

type DeltaRemote struct {
	ModelName string `yaml:"model_name"`
	Host      string `yaml:"host"`
	Port      string `yaml:"port"`
}

type DeltaInputs struct {
	Id    int    `yaml:"id"`
	Name  string `yaml:"name"`
	Shape []int  `yaml:"shape"`
	Dtype string `yaml:"dtype"`
}

type DeltaOutputs struct {
	Id    int    `yaml:"id"`
	Name  string `yaml:"name"`
	Dtype string `yaml:"dtype"`
}

type DeltaRunTime struct {
	NumThreads string `yaml:"num_threads"`
}

var DeltaConf DeltaConfig

func SetConfPath(confPath string) {
	ymlFile, err := ioutil.ReadFile(confPath)
	if err != nil {
		glog.Fatalf("read the confg file err! %s", err.Error())
	}
	err = yaml.Unmarshal(ymlFile, &DeltaConf)
	if err != nil {
		glog.Fatalf("the config file is not yaml format %s", err.Error())
	}
}
