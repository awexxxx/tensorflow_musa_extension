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

#include "mu/graph_fusion/prod_norm_act_fusion.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "mu/optimizer/graph_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace musa_fusion {

namespace {

bool IsOp(const NodeDef& node, const std::string& op_type) {
  return node.op() == op_type;
}

const std::unordered_set<std::string> kSupportedActivations = {
    "Tanh", "Sigmoid", "Log"};

const NodeDef* FindProducer(const GraphDef& graph, const std::string& input) {
  std::string name = FusionGraphUtils::GetProducerNodeName(input);
  if (name.empty()) return nullptr;
  return FusionGraphUtils::GetNodeByName(graph, name);
}

const NodeDef* ResolveIdentityLike(const GraphDef& graph, const NodeDef* node) {
  const NodeDef* current = node;
  while (current && IsOp(*current, "Identity") && current->input_size() > 0) {
    current = FindProducer(graph, current->input(0));
  }
  return current;
}

const NodeDef* FindResolvedProducer(const GraphDef& graph,
                                    const std::string& input) {
  return ResolveIdentityLike(graph, FindProducer(graph, input));
}

std::vector<const NodeDef*> FindConsumers(const GraphDef& graph,
                                          const std::string& node_name) {
  std::vector<const NodeDef*> consumers;
  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& node = graph.node(i);
    for (int j = 0; j < node.input_size(); ++j) {
      if (FusionGraphUtils::GetProducerNodeName(node.input(j)) == node_name) {
        consumers.push_back(&node);
        break;
      }
    }
  }
  return consumers;
}

bool TensorFromConstLikeNode(const GraphDef& graph, const std::string& input,
                             Tensor* out) {
  if (!out) return false;
  const NodeDef* node = FindResolvedProducer(graph, input);
  if (!node || !IsOp(*node, "Const")) return false;

  auto value_it = node->attr().find("value");
  if (value_it == node->attr().end() || !value_it->second.has_tensor()) {
    return false;
  }
  return out->FromProto(value_it->second.tensor());
}

bool ExtractFloatScalar(const GraphDef& graph, const std::string& input,
                        float* out) {
  if (!out) return false;
  Tensor tensor;
  if (!TensorFromConstLikeNode(graph, input, &tensor) ||
      tensor.NumElements() != 1) {
    return false;
  }

  switch (tensor.dtype()) {
    case DT_FLOAT:
      *out = tensor.flat<float>()(0);
      return true;
    case DT_HALF:
      *out = static_cast<float>(tensor.flat<Eigen::half>()(0));
      return true;
    case DT_BFLOAT16:
      *out = static_cast<float>(tensor.flat<bfloat16>()(0));
      return true;
    case DT_DOUBLE:
      *out = static_cast<float>(tensor.flat<double>()(0));
      return true;
    default:
      return false;
  }
}

bool IsAxisZeroKeepDimsProd(const GraphDef& graph, const NodeDef& prod_node) {
  if (!IsOp(prod_node, "Prod") || prod_node.input_size() != 2) {
    return false;
  }

  auto keep_dims_it = prod_node.attr().find("keep_dims");
  if (keep_dims_it == prod_node.attr().end() || !keep_dims_it->second.b()) {
    return false;
  }

  Tensor axes_tensor;
  if (!TensorFromConstLikeNode(graph, prod_node.input(1), &axes_tensor) ||
      axes_tensor.NumElements() != 1) {
    return false;
  }

  if (axes_tensor.dtype() == DT_INT32) {
    return axes_tensor.flat<int32>()(0) == 0;
  }
  if (axes_tensor.dtype() == DT_INT64) {
    return axes_tensor.flat<int64>()(0) == 0;
  }
  return false;
}

std::string ToFloatString(float value) {
  std::ostringstream oss;
  oss << std::setprecision(9) << value;
  return oss.str();
}

}  // namespace

MusaProdNormActFusion::MusaProdNormActFusion() = default;

bool MusaProdNormActFusion::IsKernelAvailable() const {
  if (!kernel_checked_) {
    kernel_available_ = true;
    kernel_checked_ = true;
  }
  return kernel_available_;
}

FusionMatchResult MusaProdNormActFusion::Match(const GraphDef& graph,
                                               int start_node_idx) const {
  if (start_node_idx < 0 || start_node_idx >= graph.node_size()) {
    return FusionMatchResult{};
  }

  const NodeDef& node = graph.node(start_node_idx);
  if (!IsOp(node, "Maximum")) {
    return FusionMatchResult{};
  }

  return MatchFromMaximumNode(graph, start_node_idx);
}

FusionMatchResult MusaProdNormActFusion::MatchFromMaximumNode(
    const GraphDef& graph, int maximum_node_idx) const {
  FusionMatchResult result;
  const NodeDef& maximum_node = graph.node(maximum_node_idx);

  if (!IsOp(maximum_node, "Maximum") || maximum_node.input_size() != 2) {
    return result;
  }

  const NodeDef* prod_node = nullptr;
  const NodeDef* epsilon_node = nullptr;
  int prod_input_idx = -1;

  const NodeDef* in0 = FindResolvedProducer(graph, maximum_node.input(0));
  const NodeDef* in1 = FindResolvedProducer(graph, maximum_node.input(1));
  if (in0 && IsOp(*in0, "Prod") && in1 && IsOp(*in1, "Const")) {
    prod_node = in0;
    epsilon_node = in1;
    prod_input_idx = 0;
  } else if (in1 && IsOp(*in1, "Prod") && in0 && IsOp(*in0, "Const")) {
    prod_node = in1;
    epsilon_node = in0;
    prod_input_idx = 1;
  } else {
    return result;
  }

  if (!prod_node || !epsilon_node || !IsAxisZeroKeepDimsProd(graph, *prod_node)) {
    return result;
  }

  const NodeDef* square_node =
      FindResolvedProducer(graph, prod_node->input(0));
  if (!square_node || !IsOp(*square_node, "Square") ||
      square_node->input_size() != 1) {
    return result;
  }

  float epsilon = 0.0f;
  if (!ExtractFloatScalar(graph, maximum_node.input(1 - prod_input_idx),
                          &epsilon)) {
    return result;
  }

  auto consumers = FindConsumers(graph, maximum_node.name());
  if (consumers.size() != 1) {
    return result;
  }
  const NodeDef* act_node = consumers[0];
  if (!act_node || kSupportedActivations.count(act_node->op()) == 0 ||
      act_node->input_size() != 1) {
    return result;
  }

  if (act_node->op() == "Log" && epsilon <= 0.0f) {
    return result;
  }

  result.matched = true;
  result.matched_nodes = {&maximum_node, prod_node, square_node, act_node};
  result.captured_nodes["maximum"] = &maximum_node;
  result.captured_nodes["prod"] = prod_node;
  result.captured_nodes["square"] = square_node;
  result.captured_nodes["activation"] = act_node;
  result.captured_nodes["epsilon_const"] = epsilon_node;
  result.captured_attrs["input"] = square_node->input(0);
  result.captured_attrs["activation"] = act_node->op();
  result.captured_attrs["epsilon"] = ToFloatString(epsilon);

  VLOG(1) << "[ProdNormAct::Match] SUCCESS: maximum=" << maximum_node.name()
          << ", prod=" << prod_node->name()
          << ", square=" << square_node->name()
          << ", activation=" << act_node->name()
          << ", activation_type=" << act_node->op()
          << ", epsilon=" << epsilon;

  return result;
}

Status MusaProdNormActFusion::Apply(
    GraphDef* graph, const FusionMatchResult& match_result) const {
  if (!match_result.IsValid()) {
    return Status(error::INVALID_ARGUMENT,
                  "Invalid MusaProdNormActFusion match result");
  }

  if (!IsKernelAvailable()) {
    return Status::OK();
  }

  auto act_it = match_result.captured_nodes.find("activation");
  if (act_it == match_result.captured_nodes.end() || !act_it->second) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing activation node for MusaProdNormActFusion");
  }

  const NodeDef* activation_node = act_it->second;
  const std::string output_name = activation_node->name();
  const std::string output_device = activation_node->device();

  auto input_it = match_result.captured_attrs.find("input");
  auto epsilon_it = match_result.captured_attrs.find("epsilon");
  auto activation_attr_it = match_result.captured_attrs.find("activation");
  if (input_it == match_result.captured_attrs.end() ||
      epsilon_it == match_result.captured_attrs.end() ||
      activation_attr_it == match_result.captured_attrs.end()) {
    return Status(error::INVALID_ARGUMENT,
                  "Missing captured attrs for MusaProdNormActFusion");
  }

  float epsilon = std::stof(epsilon_it->second);
  const std::string input_name = input_it->second;
  const std::string activation_type = activation_attr_it->second;

  DataType dtype = DT_FLOAT;
  auto dtype_it = activation_node->attr().find("T");
  if (dtype_it != activation_node->attr().end()) {
    dtype = dtype_it->second.type();
  }

  int output_idx = FusionGraphUtils::FindNodeIndex(*graph, output_name);
  if (output_idx >= 0) {
    FusionGraphUtils::RemoveNode(graph, output_idx);
  }

  std::vector<std::string> removable_nodes;
  for (const char* key : {"maximum", "prod", "square", "epsilon_const"}) {
    auto node_it = match_result.captured_nodes.find(key);
    if (node_it != match_result.captured_nodes.end() && node_it->second) {
      removable_nodes.push_back(node_it->second->name());
    }
  }
  FusionGraphUtils::RemoveNodesIfUnused(graph, removable_nodes);

  NodeDef* fused = graph->add_node();
  fused->set_name(output_name);
  fused->set_op("MusaProdNormActFusion");
  fused->set_device(output_device);
  fused->add_input(input_name);

  auto* attr = fused->mutable_attr();
  (*attr)["T"].set_type(dtype);
  (*attr)["epsilon"].set_f(epsilon);
  (*attr)["activation"].set_s(activation_type);

  VLOG(1) << "[ProdNormAct::Apply] Created fused node " << output_name
          << " activation=" << activation_type << " epsilon=" << epsilon;

  return Status::OK();
}

REGISTER_FUSION_PATTERN(MusaProdNormActFusion);
REGISTER_FUSION_KERNEL(MusaProdNormActFusion, []() { return true; });

}  // namespace musa_fusion
}  // namespace grappler
}  // namespace tensorflow
