from utils.config import ModelConfig, get_small_model_config

model_config: ModelConfig = get_small_model_config(
    model_config_name='seq_large_custom_tokenization',
    model_name='SeqLabelerBatched'
)