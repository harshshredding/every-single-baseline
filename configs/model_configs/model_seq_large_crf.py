from utils.config import ModelConfig, get_large_model_config

def create_model_config():
    model_config: ModelConfig = get_large_model_config(
        model_config_name='model_seq_large_default_crf',
        model_name='SeqLabelerDefaultCRF'
    )

    return model_config
