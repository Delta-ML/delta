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

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/config.h"
#include "core/io.h"

#include "yaml-cpp/yaml.h"

// using namespace ::delta::core;

namespace delta {

// glboal object for all `config.yaml`
// path -> YAML::Node
std::unordered_map<std::string, YAML::Node> _global_config;

// BaseConfig
BaseConfig::BaseConfig(std::string path) : _file_path(path) {
  auto search = _global_config.find(_file_path);

  if (search != _global_config.end()) {
    LOG_WARN << "config path:" << _file_path << " already loaded!";
  } else {
    LOG_INFO << "config path:" << _file_path;
    try {
      _global_config[path] = YAML::LoadFile(_file_path);
      LOG_INFO << "load config success";
    } catch (const YAML::Exception& e) {
      LOG_FATAL << "Error(" << e.what() << "): read [ " << _file_path
                << " ] yaml config file failed.";
    }
  }

  _custom_ops_path = config()["model"]["custom_ops_path"].as<std::string>();
  LOG_INFO << "custom ops path:" << _custom_ops_path;
}

BaseConfig::~BaseConfig() {}

YAML::Node& BaseConfig::config() const { return _global_config[_file_path]; }
std::string BaseConfig::custom_ops_path() const { return _custom_ops_path; }

// Config
Config::Config(std::string path) : BaseConfig(path) {
  DELTA_ASSERT_OK(load_graphs());
  DELTA_ASSERT_OK(load_playlist());
}

Config::~Config() {}

DeltaStatus Config::load_graphs() {
  const YAML::Node& model_cfg = config()["model"];

  int graph_num = model_cfg["graphs"].size();
  LOG_INFO << "graph num is: " << graph_num;
  for (int i = 0; i < graph_num; ++i) {
    const YAML::Node& graph_cfg = model_cfg["graphs"][i];
    std::string name = model_cfg["graphs"][i]["name"].as<std::string>();
    LOG_INFO << "Create [" << name << "] graph";
    _graphs.insert(std::pair<std::string, Graph>(name, Graph(graph_cfg)));
  }

  return DeltaStatus::STATUS_OK;
}

DeltaStatus Config::load_playlist() { return DeltaStatus::STATUS_OK; }

std::unordered_map<std::string, Graph>& Config::graphs() { return _graphs; }

// RuntimeConfig
RuntimeConfig::RuntimeConfig(std::string path) : Config(path) {
  DELTA_ASSERT_OK(load_runtime());
  _rt_cfg = config()["runtime"];
  _num_threads = _rt_cfg["num_threads"].as<int>();
  LOG_WARN << "_num_threads is " << _num_threads;
}

RuntimeConfig::~RuntimeConfig() {}

// read runtime config, e.g. threads, memeory, tf session config
DeltaStatus RuntimeConfig::load_runtime() { return DeltaStatus::STATUS_OK; }

const int RuntimeConfig::num_threads() const { return _num_threads; }

}  // namespace delta
