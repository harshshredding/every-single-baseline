from utils.config import PreprocessorConfig
from annotators import get_chatgpt_disease_annotator, get_chatgpt_per_sample_disease_annotator
import sys

def config_socialdisner_vanilla_chatgpt_external() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.socialdisner_preprocessor.PreprocessSocialDisNer',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_chatgpt_disease_annotator()]
        }
    )


def config_socialdisner_vanilla_per_sample_chatgpt_external() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.socialdisner_preprocessor.PreprocessSocialDisNer',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_chatgpt_per_sample_disease_annotator()]
        }
    )
