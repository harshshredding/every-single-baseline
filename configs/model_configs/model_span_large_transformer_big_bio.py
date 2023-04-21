from utils.config import ModelConfig, get_large_model_config_bio

def create_model_config(model_config_name: str):
    model_config: ModelConfig = get_large_model_config_bio(
        model_config_name=model_config_name,
        model_name='SpanDefaultTransformerBigger'
    )
    return model_config
