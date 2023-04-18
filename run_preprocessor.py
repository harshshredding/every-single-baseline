from utils.preprocess import PreprocessorRunType, preprocess
from structs import DatasetSplit
from all_preprocessor_configs import config_meta_special_tokens_super_gold_test_behaviour_training

preprocessor_config = config_meta_special_tokens_super_gold_test_behaviour_training()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.dry_run,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
