import utils.dropbox as dropbox_util
import subprocess
from preamble import *

# zip preprocessed data
subprocess.run(['zip','-r','preprocessed_data.zip','preprocessed_data'])
# upload zipped file to dropbox
dropbox_util.upload_big_file_cleaner('./preprocessed_data.zip', '/preprocessed_data.zip')