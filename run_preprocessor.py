from preprocess import preprocess_train_and_valid_with_window, PreprocessorRunType
from preamble import *
from structs import Dataset

# util.delete_preprocessed_data_folder()
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalJudgement')
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalPreamble')
# preprocess_train_and_valid_data('preprocessor_cdr', 'PreprocessCDR')
# preprocess_train_and_valid_with_window('livingner_preprocessor', 'PreprocessLivingNER')
preprocess_train_and_valid_with_window(
    preprocessor_module_name='preprocessor_chemdner',
    preprocessor_name='PreprocessChemD',
    preprocessor_type='window_stride_longer',
    dataset=Dataset.chem_drug_ner,
    run_mode=PreprocessorRunType.production,
    window_size=200,
    stride_size=100
)
