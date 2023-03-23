from utils.config import ModelConfig, get_large_model_config

model_config: ModelConfig = get_large_model_config(
    model_config_name='span_large_custom_tokenization',
    model_name='SpanBertCustomTokenizationBatched'
)
