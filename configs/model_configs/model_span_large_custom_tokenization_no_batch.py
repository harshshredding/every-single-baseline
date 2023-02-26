from utils.config import ModelConfig, get_large_model_config

model_config: ModelConfig = get_large_model_config(
    model_config_name='span_large_custom_tokenization_no_batch',
    model_name='SpanBert'
)

model_config.batch_size = 1
