import json
import logging
import joblib
from pathlib import Path

import pandas as pd

from scripts.preprocess import preprocess

MODEL_VERSION = 211212
API_VERSION = 211213

logger = logging.getLogger('api')
model = joblib.load(Path(f'model/{MODEL_VERSION}.pkl'))
logger.debug('Model loaded.')


def predict(request):
    """
    Get fake probability for each user requested through API

    Parameters:
        request (dict): log of queried users

    Returns:
        response (dict): API response (see readme) containing predictions and metadata
    """
    # format log in request
    try:
        df = pd.DataFrame(request['log'], columns=['UserId', 'Event', 'Category'])
    except:
        return {"error": "Invalid log format."}
    logger.info(f'Data received.')

    # preprocess
    df = preprocess(df, train_mode=False, disable_log=True)
    X = df.drop(['UserId'], axis=1)

    # add missing feature names (if any) from model/{model_name}.json
    with open(Path(f'model/{MODEL_VERSION}.json')) as json_file:
        features = json.load(json_file)['features']
    for feature in features:
        if feature not in X:
            X[feature] = 0

    # predict
    y_predicted = model.predict_proba(X)
    df['is_fake_probability'] = y_predicted[:, 1]  # get only positive probability
    df['is_fake_probability'] = df['is_fake_probability'].map('{:,.5f}'.format)  # 5 float points
    df = df[['UserId', 'is_fake_probability']]
    logger.debug(f'Predicted fake probabilty for {len(y_predicted)} users.')

    # response
    return {
        "predictions": df.to_dict('records'),
        "metadata": {
            "model_version": MODEL_VERSION,
            "api_version": API_VERSION
        }
    }
