# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""End-to-end tests for Square -> Prod -> Maximum -> Activation fusion."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rw = config.graph_options.rewrite_options
    rw.min_graph_nodes = -1
    rw.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
    rw.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
    rw.shape_optimization = rewriter_config_pb2.RewriterConfig.OFF

    custom_opt = rw.custom_optimizers.add()
    custom_opt.name = "musa_graph_optimizer"
    rw.optimizers.extend(["musa_graph_optimizer"])
    return config


def has_fused_op(partition_graphs, op_name="MusaProdNormActFusion"):
    for graph_def in partition_graphs:
        for node in graph_def.node:
            if node.op == op_name:
                return True
    return False


def get_fused_nodes(partition_graphs, op_name="MusaProdNormActFusion"):
    return [
        node
        for graph_def in partition_graphs
        for node in graph_def.node
        if node.op == op_name
    ]


def reference_prod_norm_act(x_np, epsilon, activation):
    prod = np.prod(np.square(x_np.astype(np.float64)), axis=0, keepdims=True)
    clipped = np.maximum(prod, epsilon)
    if activation == "Tanh":
        result = np.tanh(clipped)
    elif activation == "Sigmoid":
        result = 1.0 / (1.0 + np.exp(-clipped))
    elif activation == "Log":
        result = np.log(clipped)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return result.astype(np.float32)


class ProdNormActFusionTest(MUSATestCase):
    def _build_graph(self, activation, axis=0):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(tf.float32, shape=[3, 4], name="input")
                square = tf.square(x, name="square")
                prod = tf.raw_ops.Prod(
                    input=square,
                    axis=tf.constant([axis], dtype=tf.int32, name="axis"),
                    keep_dims=True,
                    name="prod",
                )
                maximum = tf.maximum(
                    prod,
                    tf.constant(1e-4, dtype=tf.float32, name="epsilon"),
                    name="maximum",
                )

                if activation == "Tanh":
                    output = tf.math.tanh(maximum, name="activation")
                elif activation == "Sigmoid":
                    output = tf.math.sigmoid(maximum, name="activation")
                elif activation == "Log":
                    output = tf.math.log(maximum, name="activation")
                else:
                    raise ValueError(f"Unsupported activation: {activation}")

        return graph, output

    def _run_graph(self, graph, output_tensor, x_np):
        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            input_tensor = graph.get_tensor_by_name("input:0")
            result = sess.run(
                output_tensor,
                feed_dict={input_tensor: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        return result, run_metadata.partition_graphs

    def test_fusion_is_applied_for_tanh(self):
        x_np = np.array(
            [[0.8, 1.2, 0.9, 1.1], [1.1, 0.7, 1.0, 0.95], [0.6, 1.3, 1.05, 0.85]],
            dtype=np.float32,
        )
        graph, output = self._build_graph("Tanh")
        result, partition_graphs = self._run_graph(graph, output, x_np)

        self.assertTrue(
            has_fused_op(partition_graphs),
            "Expected Square -> Prod -> Maximum -> Tanh to fuse into MusaProdNormActFusion",
        )
        self.assertAllClose(result, reference_prod_norm_act(x_np, 1e-4, "Tanh"))

    def test_fusion_is_applied_for_sigmoid(self):
        x_np = np.array(
            [[0.75, 0.95, 1.1, 0.8], [1.05, 0.85, 0.92, 1.2], [0.9, 1.15, 0.88, 0.78]],
            dtype=np.float32,
        )
        graph, output = self._build_graph("Sigmoid")
        result, partition_graphs = self._run_graph(graph, output, x_np)

        fused_nodes = get_fused_nodes(partition_graphs)
        self.assertTrue(fused_nodes, "Expected MusaProdNormActFusion node in optimized graph")
        # self.assertEqual(fused_nodes[0].attr["activation"].s().decode("utf-8"), "Sigmoid")
        self.assertAllClose(result, reference_prod_norm_act(x_np, 1e-4, "Sigmoid"))

    def test_fusion_is_applied_for_log(self):
        x_np = np.array(
            [[1.2, 0.8, 1.1, 0.7], [0.9, 1.05, 0.95, 1.3], [0.85, 1.1, 0.9, 1.2]],
            dtype=np.float32,
        )
        graph, output = self._build_graph("Log")
        result, partition_graphs = self._run_graph(graph, output, x_np)

        self.assertTrue(has_fused_op(partition_graphs))
        self.assertAllClose(result, reference_prod_norm_act(x_np, 1e-4, "Log"))

    def test_axis_one_pattern_is_not_fused(self):
        x_np = np.array(
            [[0.9, 1.1, 0.8, 1.05], [1.0, 0.95, 1.2, 0.85], [0.88, 1.12, 0.91, 1.04]],
            dtype=np.float32,
        )
        graph, output = self._build_graph("Tanh", axis=1)
        result, partition_graphs = self._run_graph(graph, output, x_np)

        expected = np.tanh(np.maximum(np.prod(np.square(x_np), axis=1, keepdims=True), 1e-4))
        self.assertFalse(
            has_fused_op(partition_graphs),
            "Axis=1 should not match the specialized axis=0 fusion rule",
        )
        self.assertAllClose(result, expected.astype(np.float32))


if __name__ == "__main__":
    tf.test.main()
