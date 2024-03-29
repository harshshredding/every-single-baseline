from utils.config import ModelConfig, get_large_model_config_bio

def create_model_config(model_config_module_name):
    model_config: ModelConfig = get_large_model_config_bio(
        model_config_name=model_config_module_name,
        model_name='SeqLabelerNoTokenization'
    )
    return model_config
