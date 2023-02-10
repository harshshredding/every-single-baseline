from preprocess import preprocess_train_and_valid_data
# util.delete_preprocessed_data_folder()
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalJudgement')
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalPreamble')
# preprocess_train_and_valid_data('genia_preprocessor', 'PreprocessGenia')
preprocess_train_and_valid_data('livingner_preprocessor', 'PreprocessLivingNER')
