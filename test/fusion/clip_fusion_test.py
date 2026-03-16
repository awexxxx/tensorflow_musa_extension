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
"""End-to-end tests for clip-pattern -> MusaClip fusion."""

import os
import tempfile

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase
from tensorflow.core.protobuf import config_pb2


def create_config_with_musa_optimizer():
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True

    rewriter_config = config.graph_options.rewrite_options
    custom_optimizer = rewriter_config.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    rewriter_config.min_graph_nodes = -1
    rewriter_config.optimizers.extend(["musa_graph_optimizer"])

    return config


def get_musa_clip_fused_nodes(run_metadata):
    return [
        node
        for partition_graph in run_metadata.partition_graphs
        for node in partition_graph.node
        if node.op == "MusaClip"
    ]


class ClipFusionE2ETest(MUSATestCase):
    """Tests for graph-level clip fusion."""

    def test_clip_fusion_minimum_then_maximum_is_applied(self):
        x_np = np.array(
            [[-3.0, -1.0, 2.0, 8.0],
             [0.5, 1.5, 7.0, 9.0]],
            dtype=np.float32,
        )
        lo_np = np.float32(0.0)
        hi_np = np.float32(6.0)
        expected = np.maximum(np.minimum(np.sqrt(np.abs(x_np) + 1.0), hi_np), lo_np)
        maximum_name = "fwffm_pbp_mlp/pln1_follow/clip_by_value"
        minimum_name = "fwffm_pbp_mlp/pln1_follow/clip_by_value/Minimum"

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "clip_fusion_matches.log")
            previous_log_path = os.environ.get("MUSA_CLIP_FUSION_LOG_PATH")
            os.environ["MUSA_CLIP_FUSION_LOG_PATH"] = log_path

            try:
                graph = tf.Graph()
                with graph.as_default():
                    with tf.device("/device:MUSA:0"):
                        x = tf.compat.v1.placeholder(
                            tf.float32,
                            shape=[None, 4],
                            name="fwffm_pbp_mlp/pln1_follow/input",
                        )
                        lo = tf.constant(
                            lo_np,
                            dtype=tf.float32,
                            name="fwffm_pbp_mlp/pln1_follow/clip_by_value/y",
                        )
                        hi = tf.constant(
                            hi_np,
                            dtype=tf.float32,
                            name="fwffm_pbp_mlp/pln1_follow/clip_by_value/Minimum/y",
                        )
                        sqrt_input = tf.add(
                            tf.abs(x),
                            tf.constant(1.0, dtype=tf.float32),
                            name="fwffm_pbp_mlp/pln1_follow/SqrtShift",
                        )
                        sqrt_output = tf.sqrt(
                            sqrt_input,
                            name="fwffm_pbp_mlp/pln1_follow/Sqrt",
                        )

                        output = tf.maximum(
                            tf.minimum(sqrt_output, hi, name=minimum_name),
                            lo,
                            name=maximum_name,
                        )

                graph_def = graph.as_graph_def()
                nodes_by_name = {node.name: node for node in graph_def.node}
                self.assertEqual(nodes_by_name[maximum_name].op, "Maximum")
                self.assertEqual(nodes_by_name[minimum_name].op, "Minimum")

                config = create_config_with_musa_optimizer()
                run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
                run_metadata = tf.compat.v1.RunMetadata()

                with tf.compat.v1.Session(graph=graph, config=config) as sess:
                    result = sess.run(
                        output,
                        feed_dict={x: x_np},
                        options=run_options,
                        run_metadata=run_metadata,
                    )

                fused_nodes = get_musa_clip_fused_nodes(run_metadata)

                self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
                self.assertTrue(
                    fused_nodes,
                    "Expected Maximum(Minimum(x, hi), lo) chain to be fused into MusaClip",
                )
                self.assertTrue(
                    any(node.name == maximum_name for node in fused_nodes),
                    f"Expected fused MusaClip node named {maximum_name}",
                )

                with open(log_path, "r", encoding="utf-8") as log_file:
                    log_content = log_file.read()

                self.assertIn(f"maximum_candidate={maximum_name}", log_content)
                self.assertIn(f"maximum_matched={maximum_name}", log_content)
                self.assertIn(f"inner_minimum={minimum_name}", log_content)
            finally:
                if previous_log_path is None:
                    os.environ.pop("MUSA_CLIP_FUSION_LOG_PATH", None)
                else:
                    os.environ["MUSA_CLIP_FUSION_LOG_PATH"] = previous_log_path

    def test_clip_fusion_maximum_then_minimum_is_not_applied(self):
        x_np = np.array(
            [[-3.0, -1.0, 2.0, 8.0],
             [0.5, 1.5, 7.0, 9.0]],
            dtype=np.float32,
        )
        lo_np = np.float32(0.0)
        hi_np = np.float32(6.0)
        expected = np.minimum(np.maximum(x_np, lo_np), hi_np)

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/device:MUSA:0"):
                x = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, 4], name="x"
                )
                lo = tf.constant(lo_np, dtype=tf.float32, name="lo")
                hi = tf.constant(hi_np, dtype=tf.float32, name="hi")

                output = tf.minimum(
                    tf.maximum(x, lo, name="clip_max_first"),
                    hi,
                    name="clip_min_second",
                )

        config = create_config_with_musa_optimizer()
        run_options = tf.compat.v1.RunOptions(output_partition_graphs=True)
        run_metadata = tf.compat.v1.RunMetadata()

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            result = sess.run(
                output,
                feed_dict={x: x_np},
                options=run_options,
                run_metadata=run_metadata,
            )

        fused_nodes = get_musa_clip_fused_nodes(run_metadata)

        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(
            fused_nodes,
            "Did not expect Minimum(Maximum(x, lo), hi) chain to match the simplified fusion rule",
        )


if __name__ == "__main__":
    tf.test.main()
