from utils.config import PreprocessorConfig
from annotators import get_chatgpt_disease_annotator, get_chatgpt_per_sample_disease_annotator, get_sentence_annotator,\
        get_umls_disease_annotator_exact, get_umls_disease_annotator_lowered_exact,\
        get_umls_disease_annotator_lowered_exact_word_boundaries, get_bigger_sliding_window_annotator,\
        get_umls_disease_smart_exact_word_boundaries_annotator,\
        get_sentence_annotator
        
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


def config_socialdisner_umls_exact() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.socialdisner_preprocessor.PreprocessSocialDisNer',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_umls_disease_annotator_exact()]
        }
    )


def config_socialdisner_umls_lowered_exact() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.socialdisner_preprocessor.PreprocessSocialDisNer',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_umls_disease_annotator_lowered_exact()]
        }
    )


def config_socialdisner_umls_lowered_exact_word_boundaries() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.socialdisner_preprocessor.PreprocessSocialDisNer',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_umls_disease_annotator_lowered_exact_word_boundaries()]
        }
    )

def config_chem_drug_window_longer() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.preprocessor_chemdner.PreprocessChemD',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_bigger_sliding_window_annotator()]
        }
    )

def config_genia() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.genia_preprocessor.PreprocessGenia',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_genia_article() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.genia_preprocessor.PreprocessGeniaArticleLevel',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_genia_article_window_longer() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.genia_preprocessor.PreprocessGeniaArticleLevel',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_bigger_sliding_window_annotator()]
        }
    )

# preprocessors.livingner_preprocessor.PreprocessLivingNER
def config_living_ner_vanilla() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.livingner_preprocessor.PreprocessLivingNER',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )

def config_ncbi_disease_vanilla() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_ncbi_disease_window_longer() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [get_bigger_sliding_window_annotator()]
        }
    )


def config_ncbi_disease_window_longer_umls_exact_word_boundaries() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_bigger_sliding_window_annotator(),
                get_umls_disease_annotator_lowered_exact_word_boundaries()
            ]
        }
    )


def config_ncbi_disease_sentence() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_sentence_annotator()
            ]
        }
    )


def config_ncbi_disease_window_longer_umls_smart_exact_word_boundaries() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_bigger_sliding_window_annotator(),
                get_umls_disease_smart_exact_word_boundaries_annotator()
            ]
        }
    )

