import json

from src.configuration import DATA_SET_TENDERS
from src.utils.get_file import full_path, create_file_and_folders_if_not_exist

TRACKER_REPORTED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/reported-offers.json")
TRACKER_OBSERVED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/observed-offers.json")
TRACKER_VIEWED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/viewed-offers.json")


def parse_tracker_ids(file_path, out_ids_file):
    tender_ids = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        tenders_stats = json.load(file)
        for stat in tenders_stats:
            tender_ids.append(stat['what'])

        tender_ids.sort()

        with open(out_ids_file, 'w') as ids_file:
            for item in tender_ids:
                ids_file.write("%s\n" % item)
    return

create_file_and_folders_if_not_exist(full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/observed_its.txt"))
parse_tracker_ids(TRACKER_OBSERVED_JSON, full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/observed_ids.txt"))
parse_tracker_ids(TRACKER_VIEWED_JSON, full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/viewed_ids.txt"))
parse_tracker_ids(TRACKER_REPORTED_JSON, full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/reported_ids.txt"))
