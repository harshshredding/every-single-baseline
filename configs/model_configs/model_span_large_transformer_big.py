from utils.config import ModelConfig, get_large_model_config

def create_model_config(model_config_name: str):
    model_config: ModelConfig = get_large_model_config(
        model_config_name=model_config_name,
        model_name='SpanDefaultTransformerBigger'
    )

    model_config.max_span_length = 16

    return model_config
