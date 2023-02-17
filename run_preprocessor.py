from preprocess import preprocess_train_and_valid_data, preprocess_train_and_valid_with_window
# util.delete_preprocessed_data_folder()
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalJudgement')
# preprocess_train_and_valid_data('legaleval_preprocessor', 'PreprocessLegalPreamble')
# preprocess_train_and_valid_data('genia_preprocessor', 'PreprocessGenia')
preprocess_train_and_valid_with_window('livingner_preprocessor', 'PreprocessLivingNER')
preprocess_train_and_valid_with_window('legaleval_preprocessor', 'PreprocessLegalPreamble')
