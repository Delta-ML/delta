### 异构高性能算子库

`Delta-infer`提供高性能的算子库，以及配套的perf工具组件，用来配合`Delta-infer Python API`的子图匹配模块，借助tensorflow的custom op设计方式，完成对图中新算子的替换，从而提升模型inference性能，具体的使用case如下<sub>我们还是使用transformer为例</sub>：

```c++
// include tensorflow 标准头文件
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"

// include delta-infer perf.h
#include "example/c++/basic/perf.h"


void perf_test() {
  	// 模型配置信息
		Perf::Config cfg;
    // 设置优化后模型
    cfg.graph = "/path/to/result.pb";
        
    cfg.input_layer = "input0:0,input1:0,input2:0,input3:0,input4:0,input5:0,input:0";
    cfg.input_layer_shape = "16,4,256:16,4,256:16,4,256:16,4,256:16,4,256:16,4,256:4,100";
    cfg.input_layer_type = "float,float,float,float,float,float,int32";
    cfg.input_layer_values = "1:1:1:1:1:1:1";
    cfg.output_layer = "import/e2e_model/Model_1/attn_decoder/LASDecoder/decoding_output:0";
    cfg.target_layer = "";
    cfg.num_threads = 1;
  	// 配置执行器Perf
    Perf infer(&cfg);

  	// warm up
    double warm_up_ms = 0.0;
    int warm_up = 10;
    for(int i=0; i<warm_up; i++) {
        infer.run(warm_up_ms); // perf run
    }

  	// test
    int iteration=1000;
    double total_time_ms=0.0;

    for(int i=0; i<iteration; i++) {
        infer.run(total_time_ms); // perf run
    }

    LOG(INFO) << "Average inference timings: "
              << "Warmup: " <<  warm_up_ms / warm_up << " ms, "
              << "with no stats: " << total_time_ms / iteration << " ms ";
}

int main(int argc, const char** argv) {
    perf_test();
    return 0;
}

```

熟悉tf的朋友可能已经发现，这个使用case和tf的c++接口非常类似。Delta-infer的API几乎完全使用tensorflow的原生API，因此c++接口端的使用和tensorflow的完全相同，我们这里提供的`perf.h`也是借鉴tensorflow rep里c++性能测试case编写，因此，用户完全使用之前的c++部署方式，几乎不需要改动代码即可完成高性能的部署。

唯一需要改造的流程如下：

> 1. 配置当前代码依赖， 具体方式参考`example/c++/benchmark.cc`，可以直接把部署代码放到当前统计目录编译
> 2. 用户需要使用经过子图搜索替换后的模型（protobuf model）[Details](https://github.com/pangge/delta/blob/master/deltann/infer/docs/subgraphs.md)



#### Note

>  如果用户需要在外部编译使用delta-infer，需要注意使用delta-infer可能需要引入若干依赖：
>
> 【共享库】
>
> *  libdelta_infer.so
> *  libcustom_ops.so
> *  libtensorflow_framework.so.x
> * libpythonx.xm.so.1.0
> * _pywrap_tensorflow_internal.so
> * ... 
> * 用户可以使用ldd查看example实例benchmark执行文件的依赖
>
> 【头文件】
>
> * tensorflow 安装目录中的标准头文件路径

