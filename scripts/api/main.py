import json
import logging
import joblib

import pandas as pd

from scripts.preprocess import preprocess

MODEL_VERSION = 211212
API_VERSION = 211213

logger = logging.getLogger('api')
model = joblib.load(f'model/{MODEL_VERSION}.pkl')
logger.debug('Model loaded.')


def predict(request):
    logger.debug(f'Start model {MODEL_VERSION}.')

    # format request
    try:
        df = pd.DataFrame(request['log'], columns=['UserId', 'Event', 'Category'])
    except:
        return {"error": "Invalid log format."}
    logger.info(f'Data received.')

    # preprocess
    df = preprocess(df, train_mode=False, disable_log=True)
    X = df.drop(['UserId'], axis=1)

    # missing features (if any)
    with open(f'model/{MODEL_VERSION}.json') as json_file:
        features = json.load(json_file)['features']
    for feature in features:
        if feature not in X:
            X[feature] = 0

    # predict
    y_predicted = model.predict_proba(X)
    df['is_fake_probability'] = y_predicted[:, 1]
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