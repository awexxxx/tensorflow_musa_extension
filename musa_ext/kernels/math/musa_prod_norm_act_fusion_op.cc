/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace musa {

namespace {

constexpr int kActTanh = 0;
constexpr int kActSigmoid = 1;
constexpr int kActLog = 2;

bool ParseActivation(const std::string& activation, int* activation_kind) {
  if (!activation_kind) return false;
  if (activation == "Tanh" || activation == "tanh") {
    *activation_kind = kActTanh;
    return true;
  }
  if (activation == "Sigmoid" || activation == "sigmoid") {
    *activation_kind = kActSigmoid;
    return true;
  }
  if (activation == "Log" || activation == "log") {
    *activation_kind = kActLog;
    return true;
  }
  return false;
}

}  // namespace

template <typename T>
void LaunchProdNormAct(const T* src, T* dst, int64_t outer_dim,
                       int64_t inner_size, float epsilon, int activation_kind,
                       musaStream_t stream);

template <typename T>
class MusaProdNormActFusionOp : public MusaOpKernel {
 public:
  explicit MusaProdNormActFusionOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));

    std::string activation;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation", &activation));
    OP_REQUIRES(ctx, ParseActivation(activation, &activation_kind_),
                errors::InvalidArgument("Unsupported activation: ", activation,
                                        ". Expected one of Tanh, Sigmoid, Log."));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    OP_REQUIRES(ctx, x.dims() >= 1,
                errors::InvalidArgument("Input rank must be >= 1"));

    TensorShape output_shape = x.shape();
    output_shape.set_dim(0, 1);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &y));
    if (y->NumElements() == 0) {
      return;
    }

    const int64_t outer_dim = x.dim_size(0);
    const int64_t inner_size = y->NumElements();

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    LaunchProdNormAct<T>(x.flat<T>().data(), y->flat<T>().data(), outer_dim,
                         inner_size, epsilon_, activation_kind_, stream);
  }

 private:
  float epsilon_ = 0.0f;
  int activation_kind_ = kActTanh;
};

#define REGISTER_MUSA_PROD_NORM_ACT(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(Name("MusaProdNormActFusion")                      \
                              .Device("MUSA")                                \
                              .TypeConstraint<TYPE>("T"),                    \
                          MusaProdNormActFusionOp<TYPE>);

REGISTER_MUSA_PROD_NORM_ACT(float);
REGISTER_MUSA_PROD_NORM_ACT(Eigen::half);
REGISTER_MUSA_PROD_NORM_ACT(bfloat16);

#undef REGISTER_MUSA_PROD_NORM_ACT

}  // namespace musa

REGISTER_OP("MusaProdNormActFusion")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 1e-12")
    .Attr("activation: string = 'Tanh'")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input = c->input(0);
      if (!c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      const int rank = c->Rank(input);
      if (rank < 1) {
        return errors::InvalidArgument(
            "MusaProdNormActFusion expects rank >= 1 input");
      }

      std::vector<::tensorflow::shape_inference::DimensionHandle> dims;
      dims.reserve(rank);
      dims.push_back(c->MakeDim(1));
      for (int i = 1; i < rank; ++i) {
        dims.push_back(c->Dim(input, i));
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    });

}  // namespace tensorflow
