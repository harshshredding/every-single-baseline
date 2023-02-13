from colorama import Fore, Style
from enum import Enum
from typing import Generic, TypeVar
import torch


def print_dict(some_dict):
    for key in some_dict:
        print(key, some_dict[key])


def print_section():
    print("*" * 20)


def print_green(some_string):
    print(Fore.GREEN)
    print(some_string)
    print(Style.RESET_ALL)


def die(message):
    raise RuntimeError(message)


class OptionState(Enum):
    Something = 1
    Nothing = 2


T = TypeVar('T')


class Option(Generic[T]):
    def __init__(self, val: T):
        if val is None:
            self.state = OptionState.Nothing
        else:
            self.state = OptionState.Something
            self.value = val

    def get_value(self) -> T:
        if self.state == OptionState.Nothing:
            raise RuntimeError("Trying to access nothing")
        return self.value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)
