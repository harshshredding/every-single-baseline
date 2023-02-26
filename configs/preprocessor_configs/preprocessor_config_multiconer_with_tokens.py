from structs import Dataset
from utils.config import PreprocessorConfig


preprocessor_config = PreprocessorConfig(
    preprocessor_config_name='multiconer_tokens',
    preprocessor_class_path='preprocessors.multiconer_preprocessor.PreprocessMulticoner',
    preprocessor_class_init_params={
        'preprocessor_type': 'tokens',
        'dataset': Dataset.multiconer_fine,
        'add_token_annos': True,
        'annotators': []
    }
)
