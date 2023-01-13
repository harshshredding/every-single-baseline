import dropbox
from pathlib import Path
from typing import List
home = str(Path.home())

def get_dropbox_client():
    with open(f'{home}/drop.txt', "r") as f:
        for line in f.readlines():
            line = line.strip()
            return dropbox.Dropbox(line)

def get_dropbox_file_names() -> List[str]:
    """
    Get all file names in the dropbox root folder.
    """
    dropbox_client = get_dropbox_client()
    file_entries = dropbox_client.files_list_folder('').entries
    return [entry.name for entry in file_entries]

def upload_file(file_to_upload_path: str):
    """
    Upload the given file(ignores given local folder path) to root folder.
    Replaces the file if it already exists.
    """
    print(f"Dropbox: Uploading {file_to_upload_path}")
    dropbox_client = get_dropbox_client()
    file_name = Path(file_to_upload_path).name
    with open(file_to_upload_path, "rb") as file_to_upload:
        dropbox_client.files_upload(file_to_upload.read(), f'/{file_name}', mode=dropbox.files.WriteMode.overwrite)
    print(f"Dropbox: Successfully uploaded {file_name}")