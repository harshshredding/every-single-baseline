from utils.preprocess import PreprocessorRunType, preprocess
from structs import DatasetSplit
from all_preprocessor_configs import config_ncbi_disease_window_longer_umls_exact_word_boundaries

preprocessor_config = config_ncbi_disease_window_longer_umls_exact_word_boundaries()
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.production,
    dataset_splits=[DatasetSplit.test, DatasetSplit.valid, DatasetSplit.train]
)
