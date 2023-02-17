from utils.config import ModelConfig, get_large_model_config

model_config: ModelConfig = get_large_model_config(
    model_config_name='span_width_length_restriction',
    model_name='SpanNoTokenizationBatched'
)

model_config.max_span_length = 30
