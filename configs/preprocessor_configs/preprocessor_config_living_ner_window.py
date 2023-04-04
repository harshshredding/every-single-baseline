from utils.config import PreprocessorConfig
from annotators import SlidingWindowAnnotator

def get_preprocessor_config(preprocessor_config_name: str) -> PreprocessorConfig:
    return PreprocessorConfig(
        preprocessor_config_name=preprocessor_config_name,
        preprocessor_class_path='preprocessors.livingner_preprocessor.PreprocessLivingNER',
        preprocessor_class_init_params={
            'preprocessor_type': 'window',
            'annotators': [SlidingWindowAnnotator(window_size=100, stride=50)]
        }
    )
