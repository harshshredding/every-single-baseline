from utils.config import ModelConfig, get_large_model_config

model_config: ModelConfig = get_large_model_config(
    model_config_name='SpanBatchedNoTokenizationLargeSpanish',
    model_name='SpanBertNoTokenizationBatched'
)
