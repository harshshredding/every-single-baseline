import utils.dropbox as dropbox_util
import util
import os
import zipfile

print("downloading preprocessed data")
if os.path.isdir('./preprocessed_data'):
    util.delete_preprocessed_data_folder()
dropbox_util.download_file('/preprocessed_data.zip', './preprocessed_data.zip')

print("unzipping downloaded file")
with zipfile.ZipFile('preprocessed_data.zip', 'r') as zip_ref:
    zip_ref.extractall('./')
