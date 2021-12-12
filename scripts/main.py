import sys
import joblib

from scripts.logger import init_logger
from scripts.preprocess import preprocess
from scripts.train import load_dataset

VERSION = 211212

logger = init_logger('main')


def main(path):
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

    # predict
    y_predicted = model.predict_proba(X)
    logger.debug(f'Predicted fake probabilty for {len(y_predicted)} users.')

    # save output
    df['is_fake_probability'] = y_predicted[:, 1]
    df = df[['UserId', 'is_fake_probability']]
    output_path = f'output/predict_{path.split("/")[-1]}'
    df.to_csv(output_path, index=False)
    logger.info(f'Prediction CSV saved at \"{output_path}\"')


if __name__ == "__main__":
    path = str(sys.argv[1])
    main(path)