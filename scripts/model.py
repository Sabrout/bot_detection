import json
import logging
import warnings
from datetime import datetime

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
logger = logging.getLogger('train')


def timer(start_time=None):  # a simple timer function that I use in most scripts
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        logger.debug('Time taken: %i hours %i minutes and %s seconds.' %
                     (thour, tmin, round(tsec, 2)))


def train(
    train,
    test,
    version,
):
    # train set
    y = train['Fake'].values
    X = train.drop(['Fake', 'UserId'], axis=1)

    # test set
    y_test = test['Fake'].values
    X_test = test.drop(['Fake', 'UserId'], axis=1)

    # scoring
    scorers = {
        'precision': metrics.make_scorer(metrics.precision_score, pos_label=1, average='binary'),
        'recall': metrics.make_scorer(metrics.recall_score, pos_label=1, average='binary'),
        'f1-score': metrics.make_scorer(metrics.f1_score, pos_label=1, average='binary'),
        'fb-score': metrics.make_scorer(metrics.fbeta_score,
                                        beta=0.01,
                                        pos_label=1,
                                        average='binary')
    }

    # parameter grid
    with open('config/params.json') as json_file:
        params = json.load(json_file)
        logger.debug('Parameters configuration loaded.')
    cv = StratifiedKFold(n_splits=params['n_splits'])

    # pipeline
    model = Pipeline([('sampling', SMOTE()), ('classification', XGBClassifier())])
    grid = GridSearchCV(model,
                        params['params'],
                        cv=cv,
                        scoring=scorers['fb-score'],
                        n_jobs=None,
                        return_train_score=True,
                        verbose=2)

    # fit
    start_time = timer(None)
    logger.debug('GridSearch fitting ...')
    grid.fit(X, y)
    timer(start_time)

    # predict
    y_predicted = grid.predict(X_test)
    logger.debug('Test set predicted.')

    # save model and results
    path = f'model/{version}.pkl'
    joblib.dump(grid.best_estimator_, path, compress=1)
    logger.debug(f'Model saved at \"{path}\"')
    result = {
        'params': grid.best_params_,
        'train_dataset_size': len(X),
        'features': X.columns.values.tolist(),
        'precision': metrics.precision_score(y_test, y_predicted, average='binary', pos_label=1),
        'recall': metrics.recall_score(y_test, y_predicted, average='binary', pos_label=1),
        'fb-score': metrics.fbeta_score(y_test,
                                        y_predicted,
                                        beta=0.01,
                                        average='binary',
                                        pos_label=1),
        'confusion_matrix': metrics.confusion_matrix(y_test, y_predicted).tolist()
    }
    result_path = f'model/{version}.json'
    with open(result_path, "w") as outfile:
        json.dump(result, outfile)
    logger.debug(f'Model result saved at \"{result_path}\"')
