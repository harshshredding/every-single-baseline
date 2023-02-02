import util
import train_util
import transformers
from transformers import AutoTokenizer
from args import DRY_RUN_MODE, EXPERIMENT_NAME
import time
import numpy as np
import logging  # configured in args.py
import importlib
from utils.config import DatasetConfig, ModelConfig
from random import shuffle

experiments_module = importlib.import_module(f"experiments.{EXPERIMENT_NAME}")
experiments = experiments_module.experiments

# Setup logging
root_logger = logging.getLogger()
roots_handler = root_logger.handlers[0]
roots_handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))  # change formatting
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
logging.getLogger('dropbox').setLevel(logging.WARN)
transformers.logging.set_verbosity_error()

# -------- CREATE IMPORTANT DIRECTORIES -------
logger.info("Create folders for training")

training_results_folder_path = './training_results'

mistakes_folder_path = f'{training_results_folder_path}/mistakes'
error_visualization_folder_path = f'{training_results_folder_path}/error_visualizations'
predictions_folder_path = f'{training_results_folder_path}/predictions'
models_folder_path = f'{training_results_folder_path}/models'
performance_folder_path = f'{training_results_folder_path}/performance'
test_predictions_folder_path = f'{training_results_folder_path}/test_predictions'

# Create training-results directories
util.create_directory_structure(mistakes_folder_path)
util.create_directory_structure(error_visualization_folder_path)
util.create_directory_structure(predictions_folder_path)
util.create_directory_structure(models_folder_path)
util.create_directory_structure(performance_folder_path)
util.create_directory_structure(test_predictions_folder_path)

performance_file_path = f"{performance_folder_path}/performance_{EXPERIMENT_NAME}.csv"
train_util.create_performance_file_header(performance_file_path)

dataset_config: DatasetConfig
model_config: ModelConfig

for dataset_config, model_config in experiments:
    train_util.print_experiment_info(dataset_config, model_config, EXPERIMENT_NAME, DRY_RUN_MODE)
    dataset_name = dataset_config.dataset_name

    # -------- READ DATA ---------
    # TODO: read Samples instead of reading annos, text, tokens separately.
    logger.info("Starting to read data.")
    train_samples = train_util.get_train_samples(dataset_config)
    valid_samples = train_util.get_valid_samples(dataset_config)
    test_samples = train_util.get_test_samples(dataset_config)
    logger.info(f"num train samples: {len(train_samples)}")
    logger.info(f"num valid samples: {len(valid_samples)}")
    logger.info(f"num test samples: {len(test_samples)}")
    logger.info("finished reading data.")

    # ------ MODEL INITIALISATION --------
    logger.info("Starting model initialization.")
    bert_tokenizer = AutoTokenizer.from_pretrained(model_config.bert_model_name)
    model = train_util.prepare_model(model_config, dataset_config)
    optimizer = train_util.get_optimizer(model, model_config)
    all_types = util.get_all_types(dataset_config.types_file_path, dataset_config.num_types)
    logger.debug(f"all types\n {util.p_string(list(all_types))}")
    logger.info("Finished model initialization.")

    # verify that all label types in annotations are valid types
    train_util.check_label_types(train_samples, valid_samples, all_types)

    for epoch in range(model_config.num_epochs):
        # Don't train for more than 2 epochs while testing
        if DRY_RUN_MODE and epoch > 4:
            break

        epoch_loss = []
        # Begin Training
        logger.info(f"Train epoch {epoch}")
        train_start_time = time.time()
        model.train()
        if DRY_RUN_MODE:
            train_samples = train_samples[:10]

        shuffle(train_samples)  # shuffle samples every epoch

        # Training Loop
        for train_sample in train_samples:
            optimizer.zero_grad()
            loss, predicted_annos = model(train_sample)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().detach().numpy())
        logger.info(f"Done training epoch {epoch}")
        logger.info(
            f"Epoch {epoch} Loss : {np.array(epoch_loss).mean()}, Training Time: {str(time.time() - train_start_time)} "
            f"seconds")

        # Begin Validation
        if DRY_RUN_MODE:
            valid_samples = valid_samples[:10]
            test_samples = test_samples[:10]
        logger.info(f"Epoch {epoch} DONE!\n\n\n")

        train_util.validate(
            logger=logger,
            model=model,
            validation_samples=valid_samples,
            mistakes_folder_path=mistakes_folder_path,
            predictions_folder_path=predictions_folder_path,
            error_visualization_folder_path=error_visualization_folder_path,
            performance_file_path=performance_file_path,
            EXPERIMENT_NAME=EXPERIMENT_NAME,
            dataset_name=dataset_name,
            epoch=epoch
        )

        if ((epoch + 1) % 6) == 0:  # Test every fourth epoch
            train_util.test(
                logger=logger,
                model=model,
                test_samples=test_samples,
                test_predictions_folder_path=test_predictions_folder_path,
                experiment_name=EXPERIMENT_NAME,
                dataset_name=dataset_name,
                epoch=epoch
            )

logger.info("Experiment Finished!!")
