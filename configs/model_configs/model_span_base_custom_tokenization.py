from utils.config import ModelConfig, get_small_model_config

model_config: ModelConfig = get_small_model_config(
    model_config_name='span_large_custom_tokenization',
    model_name='SpanBertCustomTokenizationBatched'
)
