
from colorama import Fore, Style


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

