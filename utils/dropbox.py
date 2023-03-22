import time
import os
import dropbox
from pathlib import Path
from typing import List
import logging

logging.getLogger('dropbox').setLevel(logging.WARN)

home = str(Path.home())


def get_dropbox_client():
    return dropbox.Dropbox(
        app_key="t5jri4p4g1q1l92",
        app_secret="5ypbsp6nr9a366o",
        oauth2_refresh_token="XdwSHDUZYVgAAAAAAAAAAbEuMwhu2GFdTvdcm8O9oUiHoSvOza18wSWC9ej7U3W0"
    )


def upload_big_file(local_file_path: str):
    print(f"Dropbox: Uploading {local_file_path}")
    dropbox_client = get_dropbox_client()
    file_name = Path(local_file_path).name
    remote_path = f'/{file_name}'
    file_size = os.path.getsize(local_file_path)
    CHUNK_SIZE = 8 * 1024 * 1024
    since = time.time()
    with open(local_file_path, 'rb') as local_file:
        uploaded_size = 0
        upload_session_start_result = dropbox_client.files_upload_session_start(local_file.read(CHUNK_SIZE))
        cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                    offset=local_file.tell())
        commit = dropbox.files.CommitInfo(path=remote_path, mode=dropbox.files.WriteMode.overwrite)
        while local_file.tell() <= file_size:
            if ((file_size - local_file.tell()) <= CHUNK_SIZE):
                dropbox_client.files_upload_session_finish(local_file.read(CHUNK_SIZE), cursor, commit)
                time_elapsed = time.time() - since
                print('Uploaded {:.2f}%'.format(100).ljust(15) + ' --- {:.0f}m {:.0f}s'
                        .format(time_elapsed//60,time_elapsed%60).rjust(15))
                break
            else:
                dropbox_client.files_upload_session_append_v2(local_file.read(CHUNK_SIZE), cursor)
                cursor.offset = local_file.tell()
                uploaded_size += CHUNK_SIZE
                uploaded_percent = 100*uploaded_size/file_size
                time_elapsed = time.time() - since
                print('Uploaded {:.2f}%'.format(uploaded_percent).ljust(15) + ' --- {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60).rjust(15), end='\r')



def upload_big_file_cleaner(local_file_path: str, remote_file_path: str):
    # get client
    dbx = get_dropbox_client()
    file_size = os.path.getsize(local_file_path)
    # Upload 50 MB chunks at a time
    CHUNK_SIZE = 50 * 1024 * 1024
    with open(local_file_path, 'rb') as local_file:
        uploaded_size = 0
        upload_session_start_result = dbx.files_upload_session_start(local_file.read(CHUNK_SIZE))
        cursor = dropbox.files.UploadSessionCursor(
            session_id=upload_session_start_result.session_id,
            offset=local_file.tell()
        )
        commit = dropbox.files.CommitInfo(
            path=remote_file_path,
            mode=dropbox.files.WriteMode.overwrite
        )
        assert local_file.tell() <= file_size, "file should be bigger than chunk"
        print("Starting Upload.")
        while local_file.tell() <= file_size: 
            if ((file_size - local_file.tell()) <= CHUNK_SIZE):
                # Last chunk remaining, so commit
                dbx.files_upload_session_finish(
                    local_file.read(CHUNK_SIZE),
                    cursor,
                    commit
                )
                print("Done uploading !")
                break
            else:
                dbx.files_upload_session_append_v2(
                    local_file.read(CHUNK_SIZE),
                    cursor
                )
                cursor.offset = local_file.tell()
                uploaded_size += CHUNK_SIZE
                uploaded_percent = 100*uploaded_size/file_size
                print('Uploaded {:.2f}%'.format(uploaded_percent))


def verify_connection():
    """
    Panics if there is some error while connecting to the
    dropbox API.
    """
    dropbox_client = get_dropbox_client()
    dropbox_client.users_get_current_account()
    print("DROPBOX: connection verified")


def get_all_file_names_in_folder() -> List[str]:
    """
    Get all file names in the dropbox root folder.
    """
    dbx = get_dropbox_client()

    folder_path = '' # root folder
    all_files = [] # collects all files here
    has_more_files = True # because we haven't queried yet
    cursor = None # because we haven't queried yet

    while has_more_files:
        if cursor is None: # if it is our first time querying
            result = dbx.files_list_folder(folder_path)
        else:
            result = dbx.files_list_folder_continue(cursor)
        all_files.extend(result.entries)
        cursor = result.cursor
        has_more_files = result.has_more
        
    print("DROPBOX: Number of total files listed: ", len(all_files))
    return [entry.name for entry in all_files]


def get_all_performance_files() -> List[str]:
    all_file_names = get_all_file_names_in_folder()
    all_performance_files = [file_name for file_name in all_file_names if ("performance" in file_name)]
    return all_performance_files


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


def download_file(remote_path: str, local_path: str):
    """
    Downloads a file at `remote_path` in dropbox to `local_path` on the local machine.
    """
    print(f"Dropbox: downloading {remote_path}")
    dropbox_client = get_dropbox_client()
    dropbox_client.files_download_to_file(local_path, remote_path)
    print(f"Dropbox: successfully downloaded to {local_path}")

verify_connection()