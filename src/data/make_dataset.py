# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd 
import numpy as np 
from pycaret.classification import *


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    #Loading the data in pandas 
    data = pd.read_csv("../../data/raw/data.csv")

    #Removing useless col and changing dtypes
    logger.info("Preparing data for Pycaret")
    data = data.drop("customerID", axis = 1)
    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
    data.TotalCharges = data.TotalCharges.astype(float)

    #split train test
    logger.info("Splitting test/set datasets.")
    train = data.sample(frac = 0.9, random_state=545)
    test = data.drop(train.index)
    test.reset_index(inplace=True, drop = True)
    train.reset_index(inplace=True, drop = True)

    #Exporting data 
    train.to_csv("../../data/processed/train.csv")
    test.to_csv("../../data/processed/test.csv")
    logger.info("Succesfully exported dataset to processed folder.")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
