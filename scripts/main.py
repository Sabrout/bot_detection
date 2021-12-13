import json
import sys
import joblib

from scripts.logger import init_logger
from scripts.preprocess import preprocess
from scripts.train import load_dataset

VERSION = 211212

logger = init_logger('main')


def main(path):
    """
    Get fake probability for each user in a given CSV file.

    Parameters:
        path (str): path to given CSV file

    Writes:
        csv (file): prediction CSV file in /output/predict_{input_csv_name}.csv
    """
    logger.debug(f'Start model {VERSION}.')

    # load csv
    df = load_dataset(path)
    logger.info(f'CSV file loaded at \"{path}\"')

    # preprocess
    df = preprocess(df, train_mode=False)
    X = df.drop(['UserId'], axis=1)
    logger.info(f'File processed.')

    # load model
    model = joblib.load(f'model/{VERSION}.pkl')
    logger.debug('Model loaded.')

    # missing feature names (if any) from model/{model_name}.json
    with open(f'model/{VERSION}.json') as json_file:
        features = json.load(json_file)['features']
    for feature in features:
        if feature not in X:
            X[feature] = 0

    # predict
    y_predicted = model.predict_proba(X)
    logger.debug(f'Predicted fake probabilty for {len(y_predicted)} users.')

    # save output
    df['is_fake_probability'] = y_predicted[:, 1]  # get only positive probability
    df['is_fake_probability'] = df['is_fake_probability'].map('{:,.5f}'.format)  # 5 float points
    df = df[['UserId', 'is_fake_probability']]
    output_path = f'output/predict_{path.split("/")[-1]}'  # write output csv in /output/
    df.to_csv(output_path, index=False)
    logger.info(f'Prediction CSV saved at \"{output_path}\"')


if __name__ == "__main__":
    # csv file path argument in command line
    # python -m scripts.main [csv_path]
    path = str(sys.argv[1])
    main(path)