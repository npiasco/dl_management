import logging.config
import logging
import yaml
import time


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


def config_log(file, root='/home/nathan/Dev/Code/dl_management/'):
    logger.debug('Loading logging file {}'.format(root+file))
    with open(file, 'rt') as f:
        config = yaml.safe_load(f.read())
    config['handlers']['file']['filename'] = root + '.log/run_{}.log'.format(time.time())
    return config


def reconfigure(file, root):
    logger.debug('Reconfiguring logging with file {}'.format(root + file))
    config = config_log(file, root)
    logging.config.dictConfig(config)


def get_logger(name):
    return logging.getLogger(name)
