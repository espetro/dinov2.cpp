"""Lightweight structural coverage for the DINOv2 GGUF converter."""

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
    import transformers  # noqa: F401
    import gguf  # noqa: F401
except ImportError as error:
    torch = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


_TorchModule = torch.nn.Module if torch is not None else object


@unittest.skipIf(torch is None, f"converter dependencies unavailable: {_IMPORT_ERROR}")
class ConverterResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        converter_path = Path(__file__).parents[1] / "scripts" / "dinov2-to-gguf.py"
        spec = importlib.util.spec_from_file_location("dinov2_to_gguf", converter_path)
        assert spec is not None and spec.loader is not None
        cls.converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.converter)

    def test_resolves_direct_and_classifier_backbones(self) -> None:
        backbone = self._Backbone()
        self.assertIs(self.converter.resolve_dinov2_backbone(backbone), backbone)
        wrapper = self._ClassifierWrapper(backbone)
        self.assertIs(self.converter.resolve_dinov2_backbone(wrapper), backbone)

    def test_rejects_unrelated_model(self) -> None:
        with self.assertRaisesRegex(AttributeError, "embeddings and encoder.layer"):
            self.converter.resolve_dinov2_backbone(torch.nn.Linear(1, 1))

    class _Encoder(_TorchModule):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList([torch.nn.Identity()])

    class _Backbone(_TorchModule):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = torch.nn.Identity()
            self.encoder = ConverterResolutionTest._Encoder()

    class _ClassifierWrapper(_TorchModule):
        def __init__(self, backbone: "ConverterResolutionTest._Backbone") -> None:
            super().__init__()
            self.backbone = backbone
            self.classifier = torch.nn.Linear(1, 1)

        @property
        def base_model(self) -> "ConverterResolutionTest._Backbone":
            return self.backbone


if __name__ == "__main__":
    unittest.main()
