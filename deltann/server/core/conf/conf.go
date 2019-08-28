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
	"delta/deltann/server/core/utils"
	"flag"
	"fmt"
	"github.com/golang/glog"
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"path/filepath"
)

type envConfig struct {
	Port string `yaml:"port"`
	Env  string `yaml:"env"`
}

type AppConf struct {
	Env envConfig `yaml:"envConfig"`
}

var AppConfig AppConf
var Profile = flag.String("profile", "develop", "deploy environment")

func init() {
	flag.Parse()

	ymlFile, err := ioutil.ReadFile(filepath.Join(utils.GetProjectPath(*Profile), fmt.Sprintf("configurations/conf.%s.yml", *Profile)))
	if err != nil {
		glog.Fatalf("read the confg file err! %s", err.Error())
	}
	err = yaml.Unmarshal(ymlFile, &AppConfig)
	if err != nil {
		glog.Fatalf("the config file is not yaml format %s", err.Error())
	}

}
