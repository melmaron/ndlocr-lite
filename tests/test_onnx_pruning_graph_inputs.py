"""Regression tests for constant pruning of graph inputs backed by initializers."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime as ort

    import onnx_batch_patcher as patcher

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit without onnx/onnxruntime
    IMPORT_ERROR = exc


def _skip_reason() -> str:
    return f"onnx/onnxruntime not available: {IMPORT_ERROR}"


def _make_identity_model() -> "onnx.ModelProto":
    """Graph input ['x', 'W'] where 'W' is also an (unused) initializer."""
    weight = numpy_helper.from_array(np.ones((3, 3), dtype=np.float32), name="W")
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["x"], ["y"], name="identity_node")],
        name="prune_graph_input_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


@unittest.skipUnless(IMPORT_ERROR is None, _skip_reason())
class PruneGraphInputInitializerTest(unittest.TestCase):
    def test_graph_input_backed_initializer_survives_pruning(self):
        model = _make_identity_model()
        self.assertEqual([init.name for init in model.graph.initializer], ["W"])

        patcher._prune_unused_constants(model)

        self.assertIn(
            "W",
            [init.name for init in model.graph.initializer],
            "initializer 'W' is also a graph input and must not be pruned",
        )
        # The dangling graph input declaration is intentionally left untouched.
        self.assertEqual([inp.name for inp in model.graph.input], ["x", "W"])

    def test_session_requires_only_x_as_feed(self):
        model = _make_identity_model()
        patcher._prune_unused_constants(model)

        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        initializer_names = {init.name for init in model.graph.initializer}
        required_feeds = sorted(
            meta.name for meta in session.get_inputs() if meta.name not in initializer_names
        )
        self.assertEqual(required_feeds, ["x"])

        # Before the fix this raised:
        # ValueError: Required inputs (['W']) are missing from input feed (['x']).
        outputs = session.run(None, {"x": np.ones((1, 3), dtype=np.float32)})
        self.assertEqual(outputs[0].shape, (1, 3))

    def test_truly_unused_initializer_is_still_pruned(self):
        weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="dead")
        graph = helper.make_graph(
            nodes=[helper.make_node("Identity", ["x"], ["y"], name="identity_node")],
            name="prune_dead_initializer_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            initializer=[weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        patcher._prune_unused_constants(model)

        self.assertEqual([init.name for init in model.graph.initializer], [])


if __name__ == "__main__":
    unittest.main()
