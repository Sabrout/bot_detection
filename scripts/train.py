from pathlib import Path

import pandas as pd

from scripts.logger import init_logger
from scripts.model import train
from scripts.preprocess import preprocess

logger = init_logger('train')


def load_dataset(path):
    """
    Load CSV un remove Unnamed columns

    Parameters:
        path (str): path to CSV file

    Returns:
        df (DataFrame): selected dataset
    """
    df = pd.read_csv(path, index_col=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def main():
    """
    Train a model using:
        version (str): name to give to the model to be trained
        train_csv (str): train dataset for the model
        test_csv (str): test dataset to evaluate the model
    """
    version = '211212'
    train_csv = Path('resources/fake_users.csv')
    test_csv = Path('resources/fake_users_test.csv')
    logger.info(f'Start {version} training.')

    # load train/test dataset
    df_train = load_dataset(train_csv)
    logger.info(f'Train dataset loaded from \"{train_csv}\"')
    df_test = load_dataset(test_csv)
    logger.info(f'Test dataset loaded from \"{test_csv}\"')

    # preprocess datasets
    df_train = preprocess(df_train, train_mode=True)
    logger.info(f'Train dataset preprocessed.')
    df_test = preprocess(df_test, train_mode=True)
    logger.info(f'Test dataset preprocessed.')

    # train model
    train(df_train, df_test, version)
    logger.info(f'Model training complete.')


if __name__ == '__main__':
    main()
