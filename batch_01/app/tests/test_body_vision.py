"""Run the shipped ONNX to catch incompatible outputs and empty-scene regressions."""
import unittest
from pathlib import Path
import numpy as np
from body_vision import PersonVision


class PersonModelTests(unittest.TestCase):
    def test_real_model_handles_landscape_portrait_and_padding(self):
        model = PersonVision(Path(__file__).resolve().parents[1])
        for height, width in ((480, 640), (640, 480), (100, 800)):
            with self.subTest(height=height, width=width):
                self.assertEqual(model.process(np.zeros((height, width, 3), np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
