import os
import logging


def logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    loggerFormat = logging.Formatter("%(asctime)s, %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(loggerFormat)
    logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(loggerFormat)
    logger.addHandler(streamHandler)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        sub_dir = args.name
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logger(os.path.join(self.save_dir, 'train.log'))
        for k, v in args.__dict__.items():
            logging.info("{}:{}".format(k, v))

    def setup(self):
        pass

    def train(self):
        pass
