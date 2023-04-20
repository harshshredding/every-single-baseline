from utils.preprocess import PreprocessorRunType, preprocess
from structs import DatasetSplit
from all_preprocessor_configs import config_ncbi_meta_all_mistakes_all_gold

preprocessor_config = config_ncbi_meta_all_mistakes_all_gold()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.production,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
