from utils.preprocess import PreprocessorRunType, preprocess
from structs import DatasetSplit
from all_preprocessor_configs import config_chem_drug_window_longer

preprocessor_config = config_chem_drug_window_longer()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.production,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
