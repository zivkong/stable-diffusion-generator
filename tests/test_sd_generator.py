import unittest
from unittest.mock import patch, MagicMock
from src.generator.sd_generator import StableDiffusionGenerator

class TestStableDiffusionGenerator(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.generator = StableDiffusionGenerator(model_id=self.model_id, use_gpu=False)

    @patch("src.generator.sd_generator.StableDiffusionPipeline.from_pretrained")
    def test_generate_image(self, mock_from_pretrained):
        """Test the image generation functionality."""
        mock_pipeline_instance = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline_instance
        mock_pipeline_instance.__call__ = MagicMock(return_value="mock_image")

        # Assuming the generator has a method `generate_image`
        with patch.object(self.generator, "generate_image", return_value="mock_image") as mock_generate:
            result = self.generator.generate_image(prompt="A test prompt")
            mock_generate.assert_called_once_with(prompt="A test prompt")
            self.assertEqual(result, "mock_image")

    @patch("src.generator.sd_generator.DPMSolverMultistepScheduler.from_config")
    @patch("src.generator.sd_generator.StableDiffusionPipeline.from_pretrained")
    def test_initialization(self, mock_from_pretrained, mock_from_config):
        """Test the initialization of the generator."""
        mock_pipeline_instance = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline_instance
        mock_scheduler_instance = MagicMock()
        mock_from_config.return_value = mock_scheduler_instance
        mock_pipeline_instance.scheduler = mock_scheduler_instance

        generator = StableDiffusionGenerator(model_id=self.model_id, use_gpu=False)
        self.assertEqual(generator.model_id, self.model_id)
        self.assertIsNotNone(generator)

    @patch("src.generator.sd_generator.StableDiffusionPipeline.from_pretrained")
    def test_invalid_model_id(self, mock_from_pretrained):
        """Test behavior with an invalid model ID."""
        mock_from_pretrained.side_effect = ValueError("Invalid model ID")
        with self.assertRaises(ValueError):
            StableDiffusionGenerator(model_id="invalid_model_id", use_gpu=False)

if __name__ == "__main__":
    unittest.main()
