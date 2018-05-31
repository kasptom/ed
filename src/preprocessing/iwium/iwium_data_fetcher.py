import json
import os

from src.configuration import DATA_SET_TENDERS
from src.preprocessing.iwium.iwium_bzb_api_client import fetch_and_save_tender_json
from src.utils.get_file import full_path, create_file_and_folders_if_not_exist

TRACKER_REPORTED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/reported-offers.json")
TRACKER_OBSERVED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/observed-offers.json")
TRACKER_VIEWED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/viewed-offers.json")

OBSERVED_FILE_PATH_IDS = full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/observed_ids.txt")
REPORTED_FILE_PATH_IDS = full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/reported_ids.txt")


def parse_tracker_ids(file_path, out_ids_file):
    tender_ids = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        tenders_stats = json.load(file)
        for stat in tenders_stats:
            if stat['what'].startswith('bzp-'):
                tender_ids.append(stat['what'])

        tender_ids = set(tender_ids)
        tender_ids = sorted(tender_ids)

        with open(out_ids_file, 'w') as ids_file:
            for item in tender_ids:
                ids_file.write("%s\n" % item)
    return


def fetch_bzp_data(ids_file_path, save_dir_name):
    json_save_dir = full_path('data/' + DATA_SET_TENDERS['label'] + '/' + save_dir_name)
    os.makedirs(json_save_dir, exist_ok=True)

    with open(ids_file_path, 'r') as ids_file:
        i = 0
        for i, l in enumerate(ids_file):
            pass
    total_number = i + 1

    with open(ids_file_path, 'r') as ids_file:
        counter = 0
        for tender_number in ids_file:
            fetch_and_save_tender_json(tender_number.strip(), json_save_dir)
            counter += 1
            print("progress {0:2.2f}%".format(counter * 100 / total_number))


# create_file_and_folders_if_not_exist(full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/observed_its.txt"))
# parse_tracker_ids(TRACKER_OBSERVED_JSON, OBSERVED_FILE_PATH_IDS)
# parse_tracker_ids(TRACKER_VIEWED_JSON, full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/viewed_ids.txt"))
# parse_tracker_ids(TRACKER_REPORTED_JSON, full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/reported_ids.txt"))

# fetch_bzp_data(OBSERVED_FILE_PATH_IDS, 'json_observed')
fetch_bzp_data(REPORTED_FILE_PATH_IDS, 'json_reported')
