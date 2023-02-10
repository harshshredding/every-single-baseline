import utils.dropbox as dropbox_util
import util
import os

if os.path.isdir('./preprocessed_data'):
    util.delete_preprocessed_data_folder()
dropbox_util.download_file('/preprocessed_data.zip', './preprocessed_data.zip')
