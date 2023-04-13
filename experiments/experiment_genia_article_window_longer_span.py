from utils.config import get_experiment_config


experiments = [
    get_experiment_config(
        model_config_module_name='model_span_large_default',
        dataset_config_name='genia_article_window_longer'
    ),
]

experiments[0].model_config.num_epochs = 20
experiments[0].model_config.max_span_length = 32
experiments[0].testing_frequency = 1
experiments[0].model_config.batch_size = 8
