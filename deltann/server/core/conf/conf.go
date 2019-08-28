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
	Port       string `yaml:"port"`
	Env        string `yaml:"env"`
}

type AppConf struct {
	Env           envConfig    `yaml:"envConfig"`
}

var AppConfig AppConf
var Profile = flag.String("profile", "develop", "deploy environment")
func init()  {
	flag.Parse()
	ymlFile, err := ioutil.ReadFile(filepath.Join(utils.GetProjectPath(*Profile), fmt.Sprintf("configurations/conf.%s.yml", *Profile)))
	if err != nil {
		glog.Fatalf("read the confg file err! %s",err.Error())
	}
	err = yaml.Unmarshal(ymlFile, &AppConfig)
	if err != nil {
		glog.Fatalf("the config file is not yaml format %s",err.Error())
	}

}
