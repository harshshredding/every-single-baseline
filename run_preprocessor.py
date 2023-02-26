from utils.preprocess import preprocess_train_and_valid_with_window, PreprocessorRunType, preprocess_vanilla, \
    get_preprocessor_class_from_path, preprocess
from preamble import *
from structs import Dataset, DatasetSplit
from utils.config import get_preprocessor_config

# util.delete_preprocessed_data_folder()
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalJudgement')
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalPreamble')
# preprocess_train_and_valid_data('preprocessor_cdr', 'PreprocessCDR')
# preprocess_train_and_valid_with_window('livingner_preprocessor', 'PreprocessLivingNER')

# preprocess_vanilla(
#     preprocessor_module_name='multiconer_preprocessor',
#     preprocessor_name='PreprocessMulticoner',
#     preprocessor_type='vanilla',
#     dataset=Dataset.multiconer_fine,
#     run_mode=PreprocessorRunType.production,
#     dataset_splits=[DatasetSplit.valid, DatasetSplit.train, DatasetSplit.test]
# )


preprocessor_config = get_preprocessor_config('preprocessor_config_multiconer_with_tokens')
preprocess(
    preprocessor_config=preprocessor_config,
    run_mode=PreprocessorRunType.production,
    dataset_splits=[DatasetSplit.valid, DatasetSplit.test, DatasetSplit.train]
)
