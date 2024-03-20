import logging.config


# Task 2
# Custom Handler
class MyHandler(logging.Handler):
    def __init__(self, filename, mode='a'):
        super().__init__()
        self.filename = filename
        self.mode = mode

    def emit(self, record: logging.LogRecord):
        message = self.format(record)
        with open(f'logs/{self.filename}', self.mode) as f:
            f.write(message + '\n')


# Task 3
# Custom Filter
class FilterByLevel(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno == self.level

