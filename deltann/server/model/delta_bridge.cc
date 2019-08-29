#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <../dpl/output/include/c_api.h>

int DeltaGoInit(char* yaml_file) {
  ModelHandel model = DeltaLoadModel(yaml_file);
  InferHandel inf = DeltaCreate(model);
    return 0;
}

int DeltaGoRun() {
int in_num = 1;
  Input ins[1];
  const char* text = "I'm angry.";
  ins[0].ptr = (void*)(&text);
  ins[0].size = 1;
  ins[0].input_name = "input_sentence";
  ins[0].graph_name = "default";

  DeltaSetInputs(inf, ins, in_num);

  DeltaRun(inf);

  int out_num = DeltaGetOutputCount(inf);
  fprintf(stderr, "The output num is %d\n", out_num);
  for (int i = 0; i < out_num; ++i) {
    int byte_size = DeltaGetOutputByteSize(inf, i);
    fprintf(stderr, "The %d output byte size is %d\n", i, byte_size);

    float* data = (float*)malloc(byte_size);
    DeltaCopyToBuffer(inf, i, (void*)data, byte_size);

    int num = byte_size / sizeof(float);
    for (int j = 0; j < num; ++j) {
      fprintf(stderr, "score is %f\n", data[j]);
    }
    free(data);
  }
    return 0;
}

int DeltaGoDestroy(){
  DeltaDestroy(inf);
  DeltaUnLoadModel(model);
  return 0;
}
