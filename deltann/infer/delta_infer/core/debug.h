#ifndef _DELTA_INFER_DEBUG_H_
#define _DELTA_INFER_DEBUG_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include<stdio.h>
#include<stdarg.h>


#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

namespace delta {

enum Level {
    Info,
    Err,
    Warn
};

template<Level level>
struct LevelTraits {
    const std::string value = "Unknown";
};

template<>
struct LevelTraits<Info> {
    const std::string value = "INFO";
};

template<>
struct LevelTraits<Err> {
    const std::string value = "ERROR";
};

template<>
struct LevelTraits<Warn> {
    const std::string value = "WARNING";
};

struct MsgEnableFlag {
public:
    static MsgEnableFlag& Instance() {
        thread_local static MsgEnableFlag _ins;
        return _ins;
    }

    void enable() {
        _value = true;
    }

    void disable() {
        _value = false;
    }

    operator bool() {
        return _value;
    }

private:
    MsgEnableFlag() {}

    bool _value{true};
};

template<Level level>
class DeltaMsg {
public:
    DeltaMsg(const char* file, unsigned line, const char* format, ...): _file(file), _line(line) {
        _ss << header_str();
    }

    explicit DeltaMsg(const char* file, unsigned line): _file(file), _line(line) {
        _ss << header_str();
    }

    template<typename T>
    DeltaMsg& operator<<(const T& msg) {
        _ss << msg;
        return *this;
    }

    // accept stream function such as std::endl and others.
    DeltaMsg<level>& operator<<(std::ostream&(*func)(std::ostream&)){
        func(_ss);
        return *this;
    }

    DeltaMsg<level>& print(const char* format, ...) {
        va_list vlist;
        va_start(vlist, format);
        char* buff = nullptr;
        int result = vasprintf(&buff, format, vlist);
        if(result == -1) {
            fprintf(stderr, "Error in format \n");
            fflush(stderr);
            abort();
        }
        _ss << buff;
        va_end(vlist);
        return *this;
    }

    ~DeltaMsg() {
#ifdef BUILD_DEBUG
        if(MsgEnableFlag::Instance()) {
            auto message = _ss.str();
            fprintf(stderr, "%s", message.c_str());
            if(level == Err) {
                fflush(stderr);
                abort();
            } else {
                fflush(stderr);
            }
        }
#endif
    }
private:
    std::string header_str() {
        std::ostringstream tmp_ss;
        tmp_ss<<LevelTraits<level>().value<<"| "<< strip_file(_file)<<":"<< _line <<"] ";
        return tmp_ss.str();
    }

    std::string strip_file(const char* file) {
        std::stringstream ss(file);
        std::string item;
        std::vector<std::string> elems;
        while (std::getline(ss, item, '/')) {
            elems.push_back(item);
        }
        return elems[elems.size()-1];
    }
private:
    const char* _file;
    unsigned _line;
    std::ostringstream _ss;
};

template<Level level>
struct Voidify {
    Voidify() {}
    void operator&(const DeltaMsg<level>&) {}

    void operator&(const std::function<void()>&) {}
};

} /* namespace delta */


#define DELTA_LOG(LEVEL) \
    delta::Voidify<LEVEL>() & delta::DeltaMsg<LEVEL>(__FILE__, __LINE__) 


#define DELTA_PRINT(FORMAT_STR, ...) \
    delta::Voidify<delta::Info>() & delta::DeltaMsg<delta::Info>(__FILE__, __LINE__)\
    .print(FORMAT_STR, __VA_ARGS__) 

#define DELTA_LOG_SCOPE_START()\
    delta::MsgEnableFlag::Instance().enable()

#define DELTA_LOG_SCOPE_END()\
    delta::MsgEnableFlag::Instance().disable()

/**
 *  // used for debug purpose
 *  DELTA_SCOPE{
 *      /// TODO
 *  }
 */
#ifdef BUILD_DEBUG
    #define DELTA_SCOPE {}
#else
    #define DELTA_SCOPE \
        delta::Voidify<delta::Info>() & [&]()
#endif

} /* namespace tensorflow */

#endif
