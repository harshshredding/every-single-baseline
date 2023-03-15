from utils.config import PreprocessorConfig

preprocessor_config = PreprocessorConfig(
    preprocessor_config_name='social_dis_ner_vanilla',
    preprocessor_class_path='preprocessors.socialdisner_preprocessor.PreprocessSocialDisNer',
    preprocessor_class_init_params={
        'preprocessor_type': 'default',
        'annotators': []
    }
)
