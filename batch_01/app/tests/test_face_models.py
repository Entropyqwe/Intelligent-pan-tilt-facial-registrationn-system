"""Real pinned ONNX smoke test; synthetic input is not an accuracy benchmark."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
from vision import FaceVision
from named_faces import NamedGallery

ROOT=Path(__file__).resolve().parents[1]


class FaceModelTests(unittest.TestCase):
    def test_real_yunet_sface_and_named_gallery_agree_on_feature_shape(self):
        model=FaceVision(ROOT,threads=2)
        with tempfile.TemporaryDirectory() as tmp:
            gallery=NamedGallery(Path(tmp)/'g.json')
            self.assertEqual(model.process(np.zeros((480,640,3),np.uint8),gallery),[])
            # Exercise the actual SFace graph and the persisted feature contract.
            # This synthetic crop tests the pipeline, never someone's identity.
            crop=np.random.default_rng(5).integers(0,256,(112,112,3),dtype=np.uint8)
            feature=model.recognizer.feature(crop)
            self.assertEqual(feature.shape,(1,128))
            pid=gallery.add('合成测试',[feature])
            self.assertEqual(gallery.match(feature)[0],pid)
            self.assertEqual(NamedGallery(gallery.path).match(feature)[0],pid)


if __name__=='__main__':unittest.main()
