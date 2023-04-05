from utils.preprocess import PreprocessorRunType, preprocess
from structs import DatasetSplit
from utils.config import get_preprocessor_config
from all_preprocessor_configs import config_socialdisner_vanilla_chatgpt_external


preprocessor_config = config_socialdisner_vanilla_chatgpt_external()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.production,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
