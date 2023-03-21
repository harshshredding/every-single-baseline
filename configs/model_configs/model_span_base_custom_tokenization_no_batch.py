from utils.config import ModelConfig, get_small_model_config

model_config: ModelConfig = get_small_model_config(
    model_config_name='span_base_custom_tokenization_no_batch',
    model_name='SpanBert'
)

model_config.batch_size = 1