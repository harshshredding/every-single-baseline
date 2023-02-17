from utils.config import ModelConfig, get_small_model_config

model_config: ModelConfig = get_small_model_config(
    model_config_name='model_span_base_default',
    model_name='SpanNoTokenizationBatched'
)
