import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)
from pudb import set_trace
from tqdm import tqdm as show_progress