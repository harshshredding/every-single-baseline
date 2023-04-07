from utils.preprocess import PreprocessorRunType, preprocess
from structs import DatasetSplit
from all_preprocessor_configs import config_socialdisner_umls_exact


preprocessor_config = config_socialdisner_umls_exact()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.dry_run,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
