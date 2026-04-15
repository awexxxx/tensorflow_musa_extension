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

import os

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase


class ProdNormActFusionOpTest(MUSATestCase):
    """Unit tests for the custom MusaProdNormActFusion operator."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        plugin_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(current_dir, "..", "..", "build", "libmusa_plugin.so"),
            os.path.join(os.path.dirname(current_dir), "..", "build", "libmusa_plugin.so"),
            os.path.join(os.getcwd(), "..", "build", "libmusa_plugin.so"),
        ]

        for path in candidate_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                plugin_path = normalized_path
                break

        if plugin_path and os.path.exists(plugin_path):
            cls._musa_ops = tf.load_op_library(plugin_path)
        else:
            cls._musa_ops = None

    def _reference(self, x_np, epsilon, activation):
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

    def _run_musa_op(self, x, epsilon, activation):
        if self._musa_ops is None or not hasattr(self._musa_ops, "musa_prod_norm_act_fusion"):
            self.skipTest(
                "MusaProdNormActFusion wrapper is not available. "
                "Build the plugin and ensure REGISTER_OP(\"MusaProdNormActFusion\") is loaded."
            )

        with tf.device("/device:MUSA:0"):
            return self._musa_ops.musa_prod_norm_act_fusion(
                x=x,
                epsilon=epsilon,
                activation=activation,
            )

    def _assert_close(self, shape, dtype, activation, epsilon=1e-4):
        np_dtype = np.float32 if dtype == tf.bfloat16 else dtype.as_numpy_dtype
        rng = np.random.RandomState(42)
        x_np = rng.uniform(0.5, 1.5, size=shape).astype(np_dtype)

        x = tf.constant(x_np, dtype=dtype)
        musa_result = self._run_musa_op(x, epsilon=epsilon, activation=activation)
        reference = self._reference(x_np.astype(np.float32), epsilon, activation)

        if dtype in (tf.float16, tf.bfloat16):
            musa_result = tf.cast(musa_result, tf.float32)
            rtol, atol = 2e-2, 2e-2
        else:
            rtol, atol = 1e-5, 1e-6

        self.assertAllClose(musa_result.numpy(), reference, rtol=rtol, atol=atol)

    def test_tanh_float32(self):
        self._assert_close(shape=[3, 8], dtype=tf.float32, activation="Tanh")

    def test_sigmoid_float16(self):
        self._assert_close(shape=[4, 6], dtype=tf.float16, activation="Sigmoid")

    def test_log_bfloat16(self):
        self._assert_close(shape=[5, 2, 3], dtype=tf.bfloat16, activation="Log")

    def test_single_row(self):
        self._assert_close(shape=[1, 7], dtype=tf.float32, activation="Tanh")


if __name__ == "__main__":
    tf.test.main()
