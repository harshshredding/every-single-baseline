import dropbox
from pathlib import Path
from typing import List

home = str(Path.home())


def get_dropbox_client():
    return dropbox.Dropbox(
        app_key="t5jri4p4g1q1l92",
        app_secret="5ypbsp6nr9a366o",
        oauth2_refresh_token="XdwSHDUZYVgAAAAAAAAAAbEuMwhu2GFdTvdcm8O9oUiHoSvOza18wSWC9ej7U3W0"
    )


def verify_connection():
    """
    Panics if there is some error while connecting to the
    dropbox API.
    """
    dropbox_client = get_dropbox_client()
    dropbox_client.users_get_current_account()


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
