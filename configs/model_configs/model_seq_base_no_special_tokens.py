from utils.config import ModelConfig, get_small_model_config
from pathlib import Path

current_file_name = Path(__file__).stem

model_config: ModelConfig = get_small_model_config(
    model_config_name=current_file_name,
    model_name='SeqLabelerNoTokenization'
)

model_config.use_special_bert_tokens = False
