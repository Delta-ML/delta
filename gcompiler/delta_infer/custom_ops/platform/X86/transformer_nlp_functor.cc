#include "delta_infer/custom_ops/transformer_cell_nlp_functor.h"

namespace tensorflow {

namespace x86 {} /* namespace x86 */

// using CPUDevice = Eigen::ThreadPoolDevice;

template <>
TransformerCellNLPFunctor<CPUDevice, float>::TransformerCellNLPFunctor() {}

template <>
TransformerCellNLPFunctor<CPUDevice, float>::~TransformerCellNLPFunctor() {}

template <>
void TransformerCellNLPFunctor<CPUDevice, float>::init(
    TransformerNLPParam<CPUDevice, float>& param) {}

template <>
void TransformerCellNLPFunctor<CPUDevice, float>::operator()(
    OpKernelContext* context, TransformerNLPParam<CPUDevice, float>& param) {
  printf("ccw test in CPU \n");
  exit(0);
}

} /* namespace tensorflow */
