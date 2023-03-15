from utils.config import PreprocessorConfig


preprocessor_config = PreprocessorConfig(
    preprocessor_config_name='chat_gpt_social_dis_ner',
    preprocessor_class_path='preprocessors.socialdisner_preprocessor.PreprocessSocialDisNerChatGPT',
    preprocessor_class_init_params={
        'preprocessor_type': 'chat_gpt',
        'annotators': []
    }
)
