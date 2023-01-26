import utils.dropbox as dropbox_util
import subprocess

# zip preprocessed data
subprocess.run(['zip','-r','preprocessed_data.zip','preprocessed_data'])
# upload zipped file to dropbox
dropbox_util.upload_file('./preprocessed_data.zip')