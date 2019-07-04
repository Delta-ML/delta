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

#include <gflags/gflags.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include "inference.h"

using namespace inference;

/*
 * Input NULL, output short*.
 * Need to free manually.
 */

int read_pcm_file(const char* pcm_file, short** pcm_samples,
                  int* pcm_sample_count) {
  FILE* fpFrom = fopen(pcm_file, "rb");
  if (!fpFrom) {
    std::cerr << "Failed open pcm file " << pcm_file << std::endl;
    return -1;
  }
  fseek(fpFrom, 0, SEEK_END);
  int pcm_bytes = ftell(fpFrom);
  if (*pcm_samples) {
    free(*pcm_samples);
    *pcm_samples = nullptr;
  }
  *pcm_samples = static_cast<short*>(malloc(pcm_bytes));
  rewind(fpFrom);
  *pcm_sample_count = pcm_bytes / sizeof(short);
  fread(*pcm_samples, sizeof(short), *pcm_sample_count, fpFrom);
  fclose(fpFrom);

  return 0;
}

struct ThreadArgs {
  ModelHandle model;
  std::list<std::string>* input_file_list;
  std::ofstream* output_file_ofs;
  double* total_speech_time_sec;
  double* total_decoding_user_time_sec;

  // mutex lock
  std::mutex* input_file_list_lock;
  std::mutex* decode_results_lock;
};

void* thread_runner(void* args) {
  unsigned int thread_id = (unsigned int)pthread_self();
  ThreadArgs* thread_args = static_cast<ThreadArgs*>(args);

  // 1. creat handle
  ModelHandle model = thread_args->model;
  InfHandle handle;
  if (STATUS_OK != create(model, &handle)) {
    std::cerr << "Thread ID " << thread_id
              << ", Create inference handle failed! \n";
  } else {
    std::cerr << "Thread ID " << thread_id
              << ", Create inference handle success! \n";
  }

  short* pcm_samples = nullptr;
  int pcm_sample_count = 0;

  Input in;
  Output out;

  // 2. inference
  while (true) {
    // Get a pcm file
    thread_args->input_file_list_lock->lock();
    if (thread_args->input_file_list->empty()) {
      thread_args->input_file_list_lock->unlock();
      std::cerr << "input_file_list is empty, break while \n";
      break;
    }
    std::string pcm_file = thread_args->input_file_list->back();
    thread_args->input_file_list->pop_back();
    // thread_args->input_file_list->push_front(pcm_file);
    thread_args->input_file_list_lock->unlock();

    // Load PCM file
    if (read_pcm_file(pcm_file.c_str(), &pcm_samples, &pcm_sample_count)) {
      continue;
    }

    double speech_time_sec = static_cast<double>(pcm_sample_count) / 8000;  // s

// inference
#if 1
    in.input = pcm_samples;  // emotion
    in.size.clear();
    in.size.push_back(pcm_sample_count);
#else  // drunk
#define random(x) (rand() % x)
    in.size.clear();
    in.inputs.clear();
    int sr = 8000;
    int count = 0;
    int total_count = 0;
    short* ptr = pcm_samples;
    while (total_count < pcm_sample_count) {
      count = random(sr * 10);
      if ((total_count + count) > pcm_sample_count) {
        count = pcm_sample_count - total_count;
      }
      // std::cerr << "count is " << count << std::endl;
      in.size.push_back(count);
      in.inputs.push_back(ptr);  // emotion
      ptr += count;
      total_count += count;
    }
#undef random
#endif

    timeval tv1;
    gettimeofday(&tv1, NULL);

    if (STATUS_OK != inference::inference(handle, in, out)) {
      std::cerr << "inference failed! \n";
      abort();
    }

    timeval tv2;
    gettimeofday(&tv2, NULL);
    float decoding_user_time_sec =
        static_cast<double>(tv2.tv_usec - tv1.tv_usec) / 1000000 +
        static_cast<double>(tv2.tv_sec - tv1.tv_sec);
    float RTF = decoding_user_time_sec / speech_time_sec;

    // Print result
    std::cerr << " pcm_file : " << pcm_file
              << ", speech time = " << speech_time_sec
              << "s, decoding_user_time_sec = " << decoding_user_time_sec
              << "s, RTF = " << RTF << std::endl;
    std::cerr << " pcm_file : " << pcm_file << ", confidence is "
              << out.confidence << ",time offset " << (out.results)[0]
              << std::endl;
    // if (out.confidence) {
    if (true) {
      thread_args->decode_results_lock->lock();
      std::string output_result = pcm_file + "\t" + out.results[0];
      *(thread_args->output_file_ofs) << output_result << std::endl;
      *(thread_args->total_speech_time_sec) += speech_time_sec;
      *(thread_args->total_decoding_user_time_sec) += decoding_user_time_sec;
      thread_args->decode_results_lock->unlock();
    }

    // Clear
    in.input = nullptr;
    out.results.clear();
  }  // end while

  // 3. destroy inference handle
  if (STATUS_OK != inference::destroy(&handle)) {
    std::cerr << "Thread ID " << thread_id << ", destory failed! \n";
  }

  free(pcm_samples);
}

void usage(char** argv) {
  std::cerr << "Usage: " << argv[0]
            << "model_type model_file thread_num pcm_list output_file"
            << std::endl;
}

// define
DEFINE_string(model_type, "emotion", "model_type");
DEFINE_string(model_path, "./", "model_path");
DEFINE_double(threshold, 0.65, "threshold");
DEFINE_int32(thread_num, 1, "thread num");
DEFINE_string(pcm_list, "pcm.list", "pcm_list");
DEFINE_string(out_file, "result.out", "result file");
DEFINE_bool(need_score, false, "If need origin score");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 6) {
    usage(argv);
  }

  std::cerr << "inference lib version : " << get_version() << std::endl;

  ModelConfig config;
  config.model_type = FLAGS_model_type;  //[emotion/emotion_v2/drunk]
  if (config.model_type == "emotion_v2" || config.model_type == "attention") {
    config.window = 25;  // ms
    config.step = 10;    // ms
  } else {
    config.window = 10 * 1000;  // ms
    config.step = 10 * 1000;    // ms
  }

  config.model_path = FLAGS_model_path;
  config.threshold = FLAGS_threshold;
  config.max_num_threads = FLAGS_thread_num;
  config.need_origin_score = FLAGS_need_score;
  const char* pcm_list = FLAGS_pcm_list.c_str();
  const char* out_file = FLAGS_out_file.c_str();
  int thread_num = config.max_num_threads;

  // print
  std::cerr << "model type is " << config.model_type << std::endl;
  std::cerr << "model path is " << config.model_path << std::endl;
  std::cerr << "threshold is " << config.threshold << std::endl;
  std::cerr << "max_threa_num is " << config.max_num_threads << std::endl;
  std::cerr << "window is " << config.window << std::endl;
  std::cerr << "stride is " << config.step << std::endl;
  std::cerr << "pcm list file is " << pcm_list << std::endl;

  // 0. read pcm list file
  std::ifstream pcm_list_ifs(pcm_list);
  if (!pcm_list_ifs) {
    std::cerr << "Error, pcm_list is not exit!\n";
    return -1;
  }
  std::list<std::string> pcm_file_list;
  while (true) {
    std::string pcm_file;
    pcm_list_ifs >> pcm_file;
    if (pcm_list_ifs.eof()) {
      break;
    }
    pcm_file_list.push_front(pcm_file);
  }
  std::cerr << "Number of PCM files in list: " << pcm_file_list.size()
            << std::endl;

  // Output file.
  std::ofstream output_file_ofs(out_file);

  // some mutex locks
  std::mutex pcm_file_list_thread_lock;
  std::mutex decode_results_lock;

  // 1. Load model
  ModelHandle model;
  if (STATUS_OK != load_model(config, &model)) {
    std::cerr << "Load model failed! \n";
    return -1;
  }

  // 2. Create thread arguments.
  // Time statistics
  double total_speech_time_sec = 0;
  double total_decoding_user_time_sec = 0;
  ThreadArgs args;
  args.model = model;
  args.input_file_list = &pcm_file_list;
  args.output_file_ofs = &output_file_ofs;
  args.total_speech_time_sec = &total_speech_time_sec;
  args.total_decoding_user_time_sec = &total_decoding_user_time_sec;

  // mutex lock
  args.input_file_list_lock = &pcm_file_list_thread_lock;
  args.decode_results_lock = &decode_results_lock;

  // 3. Start threads.
  std::vector<pthread_t> thread_vector;
  for (int idx = 0; idx < thread_num; idx++) {
    pthread_t thread;
    pthread_create(&thread, NULL, &thread_runner, static_cast<void*>(&args));
    thread_vector.push_back(thread);
  }

  // 4 Wait for threads to end.
  for (int idx = 0; idx < thread_num; idx++) {
    pthread_join(thread_vector[idx], NULL);
  }

  // 5. unload model
  if (STATUS_OK != unload_model(&model)) {
    std::cerr << "UnLoad model failed! \n";
    return -1;
  }

  // Calculate real time factor.
  double total_rtf = total_decoding_user_time_sec / total_speech_time_sec;
  std::cerr << "Total RTF: " << total_rtf
            << ", total speech time: " << total_speech_time_sec
            << ", total decoding user time: " << total_decoding_user_time_sec
            << std::endl;

  return 0;
}
