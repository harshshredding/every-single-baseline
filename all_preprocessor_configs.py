from utils.config import PreprocessorConfig
from annotators import SentenceAnnotatorSpanish, get_chatgpt_disease_annotator, get_chatgpt_per_sample_disease_annotator, get_sentence_annotator,\
        get_umls_disease_annotator_exact, get_umls_disease_annotator_lowered_exact,\
        get_umls_disease_annotator_lowered_exact_word_boundaries, get_bigger_sliding_window_annotator,\
        get_umls_disease_smart_exact_word_boundaries_annotator,\
        get_sentence_annotator,\
        get_sliding_sentence_annotator, get_umls_with_gold_annotator, get_multiple_sentence_annotator
from structs import Dataset 
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


def config_ncbi_disease_sentence_umls() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_sentence_annotator(),
                get_umls_disease_annotator_lowered_exact_word_boundaries()
            ]
        }
    )


def config_ncbi_sliding_sentence() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_sliding_sentence_annotator(),
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



def config_ncbi_disease_sentence_umls_with_gold() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_sentence_annotator(),
                get_umls_with_gold_annotator(vanilla_dataset_config_name='ncbi_disease_sentence')
            ]
        }
    )

def config_ncbi_meta() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDiseaseMeta',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_ncbi_meta_bigger_training() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDiseaseMetaBiggerValid',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )

def config_meta_special_tokens_bigger_training() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDiseaseMetaSpecialTokens',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_meta_special_tokens_super_training() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiSpecialWithSuperTraining',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_meta_special_tokens_super_gold_training() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiSpecialWithSuperGoldTraining',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )

def config_meta_special_tokens_super_gold_test_behaviour_training() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiSpecialWithSuperGoldTestBehaviour',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )


def config_ncbi_multiple_sentence() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiDisease',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                get_multiple_sentence_annotator(),
            ]
        }
    )


def config_ncbi_meta_all_mistakes_all_gold() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.ncbi_disease_preprocessor.PreprocessNcbiMetaAllGoldAllMistakes',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': []
        }
    )

def config_living_ner_sentence() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.livingner_preprocessor.PreprocessLivingNER',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [
                SentenceAnnotatorSpanish()
            ]
        }
    )


def config_meta_social_dis_ner() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.meta.MetaPreprocessor',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [],
            'test_files_folder_full_path': '/Users/harshverma/meta_bionlp/social_dis_ner/test/Apps/harshv_research_nlp',
            'valid_files_folder_full_path': '/Users/harshverma/meta_bionlp/social_dis_ner/training/Apps/harshv_research_nlp',
            'dataset_config_name': 'social_dis_ner_vanilla',
            'dataset': Dataset.social_dis_ner
        }
    )


def config_meta_living_ner() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.meta.MetaPreprocessor',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [],
            'test_files_folder_full_path': '/Users/harshverma/meta_bionlp/living_ner/test',
            'valid_files_folder_full_path': '/Users/harshverma/meta_bionlp/living_ner/training',
            'dataset_config_name': 'living_ner_window',
            'dataset': Dataset.living_ner
        }
    )



def config_meta_genia() -> PreprocessorConfig:
    name_of_this_function = sys._getframe().f_code.co_name
    return PreprocessorConfig(
        preprocessor_config_name=name_of_this_function,
        preprocessor_class_path='preprocessors.meta.MetaPreprocessor',
        preprocessor_class_init_params={
            'preprocessor_type': name_of_this_function,
            'annotators': [],
            'test_files_folder_full_path': '/Users/harshverma/meta_bionlp/genia/test/Apps/harshv_research_nlp',
            'valid_files_folder_full_path': '/Users/harshverma/meta_bionlp/genia/training/Apps/harshv_research_nlp',
            'dataset_config_name': 'genia_config_vanilla',
            'dataset': Dataset.genia
        }
    )
