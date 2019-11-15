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

#ifndef DELTANN_CORE_CONFIG_H_
#define DELTANN_CORE_CONFIG_H_

#include <iostream>
#include <string>
#include <unordered_map>

#include "core/graph.h"
#include "core/io.h"

#include "yaml-cpp/yaml.h"

namespace delta {

using delta::core::Graph;

// glboal object for all `config.yaml`
// path -> YAML::Node
extern std::unordered_map<std::string, YAML::Node> _global_config;

// base class to load yaml config
class BaseConfig {
 public:
  explicit BaseConfig(std::string path);
  virtual ~BaseConfig();

  YAML::Node& config() const;

  std::string custom_ops_path() const;

 private:
  std::string _file_path;
  std::string _custom_ops_path;
};

// load graphs and playlist from yaml
class Config : public BaseConfig {
 public:
  explicit Config(std::string path);
  ~Config();

  // Load graphs from config.yaml
  DeltaStatus load_graphs();

  // load playlist from config.yaml
  DeltaStatus load_playlist();

  std::unordered_map<std::string, Graph>& graphs();

  std::unordered_map<std::string, Graph> _graphs;

  // PlayList& playlist() const;

 private:
  // std::unordered_map<std::string, Graph> _graphs;
  // PlayList _playlist;
};

class RuntimeConfig : public Config {
 public:
  explicit RuntimeConfig(std::string path);
  ~RuntimeConfig();

  // read runtime config, e.g. threads, memeory, tf session config
  DeltaStatus load_runtime();
  const int num_threads() const;

 private:
  YAML::Node _rt_cfg;
  int _num_threads;
  // ...
};

}  // namespace delta

#endif  // DELTANN_CORE_CONFIG_H_
