import os
import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    try:
        os.makedirs(os.path.dirname(os.path.join('utils', log_file)), exist_ok=True)
        handler = logging.FileHandler(os.path.join('utils', log_file), mode='w')        
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger
    except Exception as error:
        print(f'{error}')
    

loggerError = setup_logger('first_logger', 'logs/first_logfile.log', level=logging.DEBUG)
logger = setup_logger('second_logger', 'logs/second_logfile.log')
