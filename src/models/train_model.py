# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd 
import numpy as np 
from pycaret.classification import *
import mlflow


def main():
    """ 
    Loads the data locates in data/preprocessed/train.csv and trains a model on it.
    The model will be saved in models/churn_pred.pkl
    """
    logger = logging.getLogger(__name__)

    #Importing the data
    logger.info("Importing data")
    train = pd.read_csv("../../data/processed/train.csv", index_col = 0)
    test = pd.read_csv("../../data/processed/test.csv", index_col = 0)
    ytest = test.Churn
    test = test.drop("Churn", axis = 1)

    #Init pycaret
    logger.info("Initializing Pycaret")
    s = setup(train, target = "Churn", silent = True, 
          log_experiment=True, log_data = True, 
          normalize = True, transformation = True,
          ignore_low_variance = True, remove_multicollinearity = True,
          multicollinearity_threshold = 0.95, 
          experiment_name = "nb_churn", log_plots=True)

    #Create model
    logger.info("Creating model")
    lr = create_model('lr')

    #Tune model
    logger.info("Tuning model, searching for best parameters")
    lr = tune_model(lr, n_iter = 1000)

    evaluate_model(lr)
    predict_model(lr)
    plot_model(lr, plot = 'feature')

    #Final
    final = finalize_model(lr)

    #Saving model
    logger.info("Saving model in models/churn_pred.pkl")
    save_model(final, "../../models/churn_pred")
    save_model(final, "churn_pred")
    logger.info("Successfully saved model")

    print(mlflow.get_tracking_uri())




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
