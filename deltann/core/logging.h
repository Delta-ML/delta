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

#ifndef DELTANN_CORE_LOGGING_H_
#define DELTANN_CORE_LOGGING_H_

#include <cstring>
#include <ctime>
#include <sstream>
#include <string>

#define DISALLOW_COPY_AND_ASSIGN(type) \
  type(const type&);                   \
  void operator=(const type&)

namespace delta {
namespace logging {

class DateLogger {
 public:
  DateLogger() {}

  const char* human_date() {
    time_t time_value = time(NULL);
    struct tm now;
    localtime_r(&time_value, &now);

    snprintf(_buffer, sizeof(_buffer), "%04d-%02d-%02d %02d:%02d:%02d",
             now.tm_year + base_year, now.tm_mon + 1, now.tm_mday, now.tm_hour,
             now.tm_min, now.tm_sec);

    return _buffer;
  }

 private:
  char _buffer[20];
  int base_year = 1900;
};

class LogMessage {
 public:
  LogMessage(const char* file, int line, std::string type) : _type(type) {
    const char* base = strrchr(file, '/');
    const char* base_file = base ? (base + 1) : file;
    _log_stream << "[" << _pretty_date.human_date() << "] " << base_file << ":"
                << line << ": " << _type.c_str() << " ";
  }
  ~LogMessage() {
    fprintf(stdout, "DELTA: %s \n", _log_stream.str().c_str());
    fflush(stdout);
  }
  virtual std::ostringstream& stream() { return _log_stream; }

 protected:
  std::ostringstream _log_stream;

 private:
  DateLogger _pretty_date;
  std::string _type;
  DISALLOW_COPY_AND_ASSIGN(LogMessage);
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line, std::string type)
      : LogMessage(file, line, type) {}

  ~LogMessageFatal() {
    fprintf(stdout, "DELTA: %s\n", stream().str().c_str());
    abort();
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(LogMessageFatal);
};

#define LOG_INFO \
  delta::logging::LogMessage(__FILE__, __LINE__, "INFO:").stream()
#define LOG_ERROR \
  delta::logging::LogMessage(__FILE__, __LINE__, "ERROR:").stream()
#define LOG_WARN \
  delta::logging::LogMessage(__FILE__, __LINE__, "WARNING:").stream()
#define LOG_FATAL \
  delta::logging::LogMessageFatal(__FILE__, __LINE__, "FATAL:").stream()

}  // namespace logging

}  // namespace delta

#define DELTA_CHECK(x) \
  if (!(x)) LOG_FATAL << "Check failed: " #x << ' '

#define DELTA_CHECK_LT(x, y) DELTA_CHECK((x) < (y))
#define DELTA_CHECK_GT(x, y) DELTA_CHECK((x) > (y))
#define DELTA_CHECK_LE(x, y) DELTA_CHECK((x) <= (y))
#define DELTA_CHECK_GE(x, y) DELTA_CHECK((x) >= (y))
#define DELTA_CHECK_EQ(x, y) DELTA_CHECK((x) == (y))
#define DELTA_CHECK_NE(x, y) DELTA_CHECK((x) != (y))

#define DELTA_ASSERT_OK(status) DELTA_CHECK_EQ(status, DeltaStatus::STATUS_OK)

#endif  // DELTANN_CORE_LOGGING_H_
