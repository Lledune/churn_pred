# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd 
import numpy as np 
from pycaret.classification import *
from pycaret.utils import check_metric
import mlflow



def main():
    """ 
    Loads the churn prediction model and make predictions on the test dataset.
    """
    logger = logging.getLogger(__name__)

    #Load model 
    logger.info('Loading the churn_pred model')

    logged_model = 'G:/Mon Drive/MlOpsProject/churn_pred/src/models/mlruns/215166899550038237/251551ce59f9485b9f137cc2571f3279/artifacts/model/model'
    #model = load_model('../../models/churn_pred')

    model = load_model(logged_model)

    #Load test data
    logger.info("Loading test data")
    test = pd.read_csv('../../data/processed/test.csv', index_col=0)
    ytest = test.Churn
    test = test.drop('Churn', axis = 1)

    #Predict 
    logger.info("Making predictions")
    preds = predict_model(model, test)

    #Metric
    acc = check_metric(ytest, preds.Label, 'Accuracy')
    precision = check_metric(ytest, preds.Label, 'Precision')
    recall = check_metric(ytest, preds.Label, 'Recall')
    logger.info(f'Accuracy : {acc}')
    logger.info(f'Precision : {precision}')
    logger.info(f'Recall : {recall}')
    
    #Save results
    preds.to_csv("./prediction_results/preds.csv")
    logger.info("Results saved in src/models/prediction_results")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
