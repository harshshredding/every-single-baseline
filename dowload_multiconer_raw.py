import utils.dropbox as dropbox_util
import zipfile

print("Downloading multiconer raw files")
dropbox_util.download_file('/multiconer-data-raw.zip', './multiconer-data-raw.zip')
print("Unzipping downloaded files")
with zipfile.ZipFile('multiconer-data-raw.zip', 'r') as zip_ref:
    zip_ref.extractall('./')
