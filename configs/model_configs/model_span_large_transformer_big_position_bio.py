from utils.config import ModelConfig, get_large_model_config_bio

def create_model_config():
    model_config: ModelConfig = get_large_model_config_bio(
        model_config_name='span_large_default',
        model_name='SpanDefaultTransformerBiggerPosition'
    )
    return model_config
