import json
import logging

import pandas as pd


def preprocess(df, train_mode=False, disable_log=False):
    """
    Prepare train/test datasets to be used by the model

    Parameters:
        df (DataFrame): data to be processed
        train_mode (bool): avoid processing (Fake) column or (y) for predictions
        disable_log (bool): to stop logging while using API (production common practice)

    Returns:
        df_encoded (DataFrame): processed data
    """
    # set logger
    if train_mode:
        logger = logging.getLogger('train')
    else:
        logger = logging.getLogger('main')
    # if not logging
    logger.disabled = disable_log

    size = len(df)
    df = df.sample(frac=1, random_state=42)  # shuffle
    logger.debug(f'Preprocessing dataset of {size} rows')

    # load predefined categories in case a category is missing
    # in training but not at runtime
    with open('config/categories.json') as json_file:
        categories = json.load(json_file)
        logger.debug('Categories configuration loaded.')

    # data formatting
    df['UserId'] = df['UserId'].astype(str)
    df['Event'] = df['Event'].astype(str)
    df['Category'] = df['Category'].astype(str)
    if train_mode:
        df = df[df['Fake'].notna()]
        df['Fake'] = (df['Fake'] > 0).astype(int)

    # clean dataframe
    if train_mode:
        df = df[['UserId', 'Event', 'Category', 'Fake']]
    else:
        df = df[['UserId', 'Event', 'Category']]
    df = df[df['Event'].isin(categories['Event'])]
    df = df[df['Category'].isin(categories['Category'])]
    df = df.sample(frac=1, random_state=42)
    logger.debug(f'Dataset cleaned. {size - len(df)} rows dropped')

    # check for missing categories
    missing = {}
    if train_mode:
        # check if all Event types exists
        df_events = df['Event'].unique()
        for event in categories['Event']:
            if not event in df_events:
                missing['Event'] = event
        # check if all Category types exists
        df_cats = df['Category'].unique()
        for cat in categories['Category']:
            if not cat in df_cats:
                missing['Category'] = cat
        if not missing:
            logger.error('Missing categories for training.')
            logger.error(str(missing))

    # encode features for each unique pair (Event, Category)
    df['features'] = df['Event'] + '_' + df['Category']
    df_encoded = pd.get_dummies(df, columns=['features'], prefix='feat')
    df_encoded = df_encoded.drop(columns=['Event', 'Category'])  # drop old columns
    logger.debug(f'Features encoded.')

    # group by users
    df_encoded = df_encoded.groupby(['UserId']).sum()  # each data row repesents a user's whole log
    if train_mode:
        df_encoded['Fake'] = (df_encoded['Fake'] > 0).astype(int)  # rebinarize [Fake]
    df_encoded.reset_index(inplace=True)
    df_encoded = df_encoded.rename(columns={'index': 'UserId'})  # restore column name for [UserId]
    logger.debug(f'Dataset grouped by Users')

    return df_encoded
