from root_dir import ROOT_DIR


def full_path(relative_file_path: str):
    if not relative_file_path.startswith("/"):
        relative_file_path = "/" + relative_file_path
    return ROOT_DIR + relative_file_path
