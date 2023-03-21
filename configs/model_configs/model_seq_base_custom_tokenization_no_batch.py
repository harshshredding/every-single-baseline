from utils.config import ModelConfig, get_small_model_config

model_config: ModelConfig = get_small_model_config(
    model_config_name='seq_base_custom_tokenization_no_batch',
    model_name='JustBert3Classes'
)

model_config.batch_size = 1