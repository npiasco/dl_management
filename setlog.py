import logging.config
import logging
import yaml
import time, os


# Default configuration
def_config = {
    'version': 1,
    'formatters': {
        'standard': {
            '()': 'colorlog.ColoredFormatter',
            'format': '%(log_color)s%(name)s - %(levelname)-8s%(reset)s %(blue)s%(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['default'],
        'level': 'DEBUG',
    }

}

logging.config.dictConfig(def_config)
logger = logging.getLogger(__name__)
logger.debug('Log set to default behaviour:')
logger.debug(str(def_config))


def config_log(conf_file, log_folder, timestamp=False):
    logger.debug('Loading logging file {}'.format(conf_file))
    with open(conf_file, 'rt') as f:
        config = yaml.safe_load(f.read())
    try:
        os.mkdir(log_folder)
    except FileExistsError:
        print('Directory {} already exist'.format(log_folder))

    if timestamp:
        config['handlers']['file']['filename'] = log_folder + 'run_{}.log'.format(time.time())
    else:
        config['handlers']['file']['filename'] = log_folder + 'run.log'
    return config


def reconfigure(conf_file, log_file):
    logger.debug('Reconfiguring logging with file {}'.format(conf_file))
    config = config_log(conf_file, log_file)
    logging.config.dictConfig(config)


def get_logger(name):
    return logging.getLogger(name)
