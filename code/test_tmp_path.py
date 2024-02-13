import pytest
from transformers import AutoModelForCausalLM


class TestTmpPath:
    @pytest.fixture
    def model(self):
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
        return model

    def test_save(self, model, tmp_path):
        model_dir = tmp_path / "bloomz"
        model.save_pretrained(model_dir)
        print(model_dir)
        assert (model_dir / "config.json").is_file()
        assert (model_dir / "model.safetensors").is_file()
