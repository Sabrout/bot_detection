import logging

import pandas as pd
from scripts.logger import init_logger
from scripts.preprocess import preprocess
from scripts.model import train

logger = init_logger('train')


def load_dataset(path):
    df = pd.read_csv(path, index_col=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def main():
    version = '211212'
    train_csv = 'resources/fake_users.csv'
    test_csv = 'resources/fake_users_test.csv'
    logger.info(f'Start {version} training.')

    # load train and test dataset
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
