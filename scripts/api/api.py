import json

from bottle import BaseRequest, HTTPError, HTTPResponse, error, get, post, request, run
from scripts.api.main import API_VERSION, MODEL_VERSION, predict
from scripts.logger import init_logger

PORT = 8090

BaseRequest.MEMFILE_MAX = 1024 * 1024 * 1024
logger = init_logger('api', log_file=False)


@get('/fake_probability')
def get_fake_probability():
    """
    Get fake probability for each user based on a 5 minute log

    Parameters:
        log (list): 2D-array of [UserId, Event, Category]

    Returns:
        response (dict): API response (see readme) containing predictions and metadata
    """
    if not request.json:
        return HTTPResponse({'Empty request was given.'}, 400)
    if not request.json.get('log', False):
        return HTTPResponse({'Key \"log\" not found'}, 400)

    return predict(request.json)


@get('/settings')
def get_settings():
    """
    Get settings for the current api
    (model version, api version)

    Returns:
        response (dict): Settings response
    """
    return json.dumps({
        'model_version': MODEL_VERSION,
        'api_version': API_VERSION,
    })


def main():
    run(host='0.0.0.0', port=PORT, server='paste')
    logger.info('API running ...')


@error(404)
def error404():
    return json.dumps({'error': 'URL not found'})


@error(500)
def error500(http_error: HTTPError):
    return json.dumps({'error': [line.strip() for line in http_error.traceback.split('\n') if len(line.strip()) > 0]})


if __name__ == '__main__':
    main()
