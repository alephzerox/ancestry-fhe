import time
from enum import Enum


class PrintMessage(Enum):
    NO = 0
    ON_START_ONLY = 1
    ON_START_AND_STOP = 2


class Stopwatch:
    def __init__(self, print_message=PrintMessage.ON_START_ONLY):
        self._started = False
        self._print_message = print_message
        self._message = None

    def start(self, message=""):
        assert not self._started

        self._started = True
        self._start_time_seconds = time.time()

        self._message = message
        if self._print_message == PrintMessage.ON_START_ONLY:
            print(message, end='', flush=True)
        elif self._print_message == PrintMessage.ON_START_AND_STOP:
            print(message)

    def stop(self):
        assert self._started

        self._started = False
        stop_time_seconds = time.time()
        self._elapsed_seconds = stop_time_seconds - self._start_time_seconds

        assert self._elapsed_seconds >= 0

        if self._print_message == PrintMessage.ON_START_ONLY:
            print(f" {self._elapsed_seconds:.2f} s")
        elif self._print_message == PrintMessage.ON_START_AND_STOP:
            print(f"'{self._message}' done in {self._elapsed_seconds:.2f} s")

        self._message = None

    @property
    def elapsed_seconds(self):
        return self._elapsed_seconds
