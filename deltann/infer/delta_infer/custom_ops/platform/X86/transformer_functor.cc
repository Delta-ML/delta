#include "delta_infer/custom_ops/transformer_cell_functor.h"

namespace tensorflow {

namespace x86 {

} /* namespace x86 */

//using CPUDevice = Eigen::ThreadPoolDevice;

template<>
TransformerCellFunctor<CPUDevice, float>::TransformerCellFunctor() {}

template<>
TransformerCellFunctor<CPUDevice, float>::~TransformerCellFunctor() {}

template<>
void TransformerCellFunctor<CPUDevice, float>::init(TransformerParam<CPUDevice, float>& param) {}

template<>
void TransformerCellFunctor<CPUDevice, float>::operator() (OpKernelContext* context, TransformerParam<CPUDevice, float>& param) {
        printf("didi test in CPU \n");
        exit(0);
}

} /* namespace tensorflow */ 

