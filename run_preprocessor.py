from utils.preprocess import PreprocessorRunType, preprocess
from structs import DatasetSplit
from utils.config import get_preprocessor_config

preprocessor_config = get_preprocessor_config('preprocessor_config_social_dis_ner')
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.dry_run,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
