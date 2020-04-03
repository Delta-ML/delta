#ifndef _DELTA_INFER_CONFIG_H_
#define _DELTA_INFER_CONFIG_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include<stdio.h>
#include<stdarg.h>

namespace tensorflow {

namespace delta {

/// basic entry value of config
class Entry {
public:
    Entry() {}
    virtual ~Entry() {};
};

class Config {
public:
    static Config& Instance() {
      static Config config_ins; 
      return config_ins;
    } 

    bool have(const std::string& key) const {
        for(auto it = _algo_map.begin(); it != _algo_map.end();++it) {
            if(it->first == key) {
                return true;
            }
        }
        return false;
    }

    void add(const std::string& key, std::shared_ptr<Entry> entry) {
        _algo_map[key] = entry;
    }
    
    std::shared_ptr<Entry> operator[](const std::string& key) {
        if(this->have(key)) {
            return _algo_map[key];
        }
        return nullptr;
    }

private:
    Config() {}
    std::unordered_map<std::string, std::shared_ptr<Entry> > _algo_map;
};


} /* namespace delta */


} /* namespace tensorflow */

#endif
