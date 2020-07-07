#include "example/c++/basic/perf.h"

namespace tensorflow {

namespace {

std::once_flag flag;

void PrintTensor(tensorflow::Tensor& output_tensor, const char* file_path) {
  std::call_once(flag, [&]() {
    printf("get into print tensor\n");
    auto output_array = output_tensor.flat<float>();
    // int* data = output_matrix.data();
    std::ofstream out(file_path, std::ios_base::binary);
    if (out.good()) {
      // std::cout << "dims? " << output_tensor.shape().dim_size(0) << "\n";
      for (int i = 0; i < 50; i++) {
        // std::cout << output_array(i) << "\n";
        // out.write((char *)&(output_array(i)), sizeof(float));
        out << output_array(i) << "\n";
      }
    }
    out.close();
  });
}

#if 1
template <class T>
void InitializeTensor(const std::vector<float>& initialization_values,
                      Tensor* input_tensor) {
  auto type_tensor = input_tensor->flat<T>();
  type_tensor = type_tensor.constant(0);
  for (int i = 0; i < type_tensor.size(); i++) {
    type_tensor(i) = (float)(i % 10 + 1);
  }
}

#else

template <class T>
void InitializeTensor(const std::vector<float>& initialization_values,
                      Tensor* input_tensor) {
  auto type_tensor = input_tensor->flat<T>();
  type_tensor = type_tensor.constant(0);
  if (!initialization_values.empty()) {
    int i = 0;
    for (; i < initialization_values.size(); ++i) {
      type_tensor(i) = static_cast<T>(initialization_values[i]);
    }
    int cur = i - 1;
    for (; i < type_tensor.size(); i++) {
      type_tensor(i) = static_cast<T>(initialization_values[cur]);
    }
  }
}
#endif

void CreateTensorsFromInputInfo(
    const std::vector<InputLayerInfo>& inputs,
    std::vector<std::pair<string, tensorflow::Tensor> >* input_tensors) {
  for (const InputLayerInfo& input : inputs) {
    Tensor input_tensor(input.data_type, input.shape);
    switch (input.data_type) {
      case DT_INT32: {
        InitializeTensor<int32>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_FLOAT: {
        InitializeTensor<float>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_QUINT8: {
        InitializeTensor<quint8>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_UINT8: {
        InitializeTensor<uint8>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_BOOL: {
        InitializeTensor<bool>(input.initialization_values, &input_tensor);
        break;
      }
      case DT_STRING: {
        if (!input.initialization_values.empty()) {
          LOG(FATAL) << "Initialization values are not supported for strings";
        }
        auto type_tensor = input_tensor.flat<string>();
        type_tensor = type_tensor.constant("");
        break;
      }
      default:
        LOG(FATAL) << "Unsupported input type: "
                   << DataTypeString(input.data_type);
    }
    input_tensors->push_back({input.name, input_tensor});
  }
}

Status InitializeSession(int num_threads, const string& graph,
                         std::unique_ptr<Session>* session,
                         std::unique_ptr<GraphDef>* graph_def) {
  LOG(INFO) << "Loading TensorFlow.";

  tensorflow::SessionOptions options;
  // set thread pool
  tensorflow::ConfigProto& config = options.config;
  config.mutable_gpu_options()->set_allow_growth(true);
  // config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
  if (num_threads > 0) {
    config.set_intra_op_parallelism_threads(1);
    config.set_inter_op_parallelism_threads(num_threads);
  }
  // set grappler ref:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/rewriter_config.proto
  /*tensorflow::RewriterConfig* rewirter_cfg =
  config.mutable_graph_options()->mutable_rewrite_options();
  tensorflow::RewriterConfig::CustomGraphOptimizer* graph_optimizer =
  rewirter_cfg->add_custom_optimizers();
  graph_optimizer->set_name("SubGraphFusionOptimizer");*/

  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session->reset(tensorflow::NewSession(options));
  graph_def->reset(new GraphDef());
  // tensorflow::GraphDef tensorflow_graph;
  Status s = ReadBinaryProto(Env::Default(), graph, graph_def->get());
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), graph, graph_def->get());
  }

  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
    return s;
  }

  s = (*session)->Create(*(graph_def->get()));
  tensorflow::graph::SetDefaultDevice("/gpu:0", graph_def->get());
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Session: " << s;
    return s;
  }

  return Status::OK();
}

}  // namespace

Perf::Perf(const Config* config) : cfg(config) {
  std::vector<string> input_layers = str_util::Split(cfg->input_layer, ',');
  std::vector<string> input_layer_shapes =
      str_util::Split(cfg->input_layer_shape, ':');
  std::vector<string> input_layer_types =
      str_util::Split(cfg->input_layer_type, ',');
  std::vector<string> input_layer_values =
      str_util::Split(cfg->input_layer_values, ':');
  outputs = str_util::Split(cfg->output_layer, ',');
  targets = str_util::Split(cfg->target_layer, ',');
  if ((input_layers.size() != input_layer_shapes.size()) ||
      (input_layers.size() != input_layer_types.size())) {
    LOG(ERROR) << "There must be the same number of items in input_layer,"
               << " input_layer_shape, and input_layer_type, for example"
               << " input_layer=input1,input2 input_layer_type=float,float "
               << " input_layer_shape=1,224,224,4:1,20";
    LOG(ERROR) << "input_layer=" << cfg->input_layer << " ("
               << input_layers.size() << " items)";
    LOG(ERROR) << "--input_layer_type=" << cfg->input_layer_type << " ("
               << input_layer_types.size() << " items)";
    LOG(ERROR) << "--input_layer_shape=" << cfg->input_layer_shape << " ("
               << input_layer_shapes.size() << " items)";
    exit(1);
  }
  const size_t inputs_count = input_layers.size();

  LOG(INFO) << "Graph: [" << cfg->graph << "]";
  // LOG(INFO) << "Init ops:" << init_ops_string;
  LOG(INFO) << "Input layers: [" << cfg->input_layer << "]";
  LOG(INFO) << "Input shapes: [" << cfg->input_layer_shape << "]";
  LOG(INFO) << "Input types: [" << cfg->input_layer_type << "]";
  LOG(INFO) << "Output layers: [" << cfg->output_layer << "]";
  LOG(INFO) << "Target layers: [" << cfg->target_layer << "]";
  LOG(INFO) << "Num threads: [" << cfg->num_threads << "]";

  // Initialize and  create graph for session from pb
  int64 initialization_start_us = Env::Default()->NowMicros();
  Status initialize_status =
      InitializeSession(cfg->num_threads, cfg->graph, &session, &graph_def);
  int64 initialization_end_us = Env::Default()->NowMicros();
  double initialization_time_ms =
      (initialization_end_us - initialization_start_us) / 1000.0;
  if (!initialize_status.ok()) {
    exit(1);
  }
  LOG(INFO) << "Graph Initialized session in " << initialization_time_ms
            << " ms";

  // Create and initialize inputs
  std::vector<InputLayerInfo> inputs;
  for (int n = 0; n < inputs_count; ++n) {
    InputLayerInfo input;
    CHECK(DataTypeFromString(input_layer_types[n], &input.data_type))
        << input_layer_types[n] << " [idx " << n << "] was an invalid type";
    std::vector<int32> sizes;
    CHECK(str_util::SplitAndParseAsInts(input_layer_shapes[n], ',', &sizes))
        << "Incorrect size string specified: " << input_layer_shapes[n];
    for (int i = 0; i < sizes.size(); ++i) {
      int32 size = sizes[i];
      if (size == -1) {
        LOG(ERROR) << "Any unknown sizes in the shapes (-1's) must be replaced"
                   << " with the size you want to benchmark with.";
        exit(1);
      }
      input.shape.AddDim(sizes[i]);
    }
    input.name = input_layers[n];
    // printf("input name %s\n", input.name.c_str());
    if (n < input_layer_values.size()) {
      CHECK(str_util::SplitAndParseAsFloats(input_layer_values[n], ',',
                                            &input.initialization_values))
          << "Incorrect initialization values string specified: "
          << input_layer_values[n];
    }
    inputs.push_back(input);
  }
  CreateTensorsFromInputInfo(inputs, &input_tensors);
}

void Perf::with_stat(const Perf::StatConfig* stat_cfg = nullptr) {
  if (stat_cfg == nullptr) {
    return;
  }
  // set stat option
  StatSummarizerOptions stats_options;
  stats_options.show_run_order = stat_cfg->show_run_order;
  stats_options.run_order_limit = stat_cfg->run_order_limit;
  stats_options.show_time = stat_cfg->show_time;
  stats_options.time_limit = stat_cfg->time_limit;
  stats_options.show_memory = stat_cfg->show_memory;
  stats_options.memory_limit = stat_cfg->memory_limit;
  stats_options.show_type = stat_cfg->show_type;
  stats_options.show_summary = stat_cfg->show_summary;
  stats.reset(new tensorflow::StatSummarizer(stats_options));

  // set run options
  run_options.set_trace_level(RunOptions::FULL_TRACE);
}

Status Perf::run(double& total_time_ms) {
  std::vector<tensorflow::Tensor> output_tensors;

  const int64 start_time = Env::Default()->NowMicros();
  Status s = session->Run(run_options, input_tensors, outputs, targets,
                          &output_tensors, &run_metadata);
  const int64 end_time = Env::Default()->NowMicros();
  total_time_ms += (end_time - start_time) / 1000.0;

  // run only once
  PrintTensor(output_tensors[0], "./right.txt");

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return s;
  }

  if (stats != nullptr) {
    assert(run_metadata.has_step_stats());
    const StepStats& step_stats = run_metadata.step_stats();
    stats->ProcessStepStats(step_stats);
  }
  return Status::OK();
}

}  // namespace tensorflow
