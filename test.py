import utils.dropbox as dropbox_util
import subprocess

def throws():
    raise StopIteration

try:
    throws()
except StopIteration:
    print("Caught StopIteration !")