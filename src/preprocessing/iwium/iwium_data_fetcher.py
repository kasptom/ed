import json
import os

from src.configuration import DATA_SET_TENDERS
from src.preprocessing.iwium.iwium_bzb_api_client import fetch_data_daily
from src.utils.get_file import full_path

TRACKER_REPORTED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/reported-offers.json")
TRACKER_OBSERVED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/observed-offers.json")
TRACKER_VIEWED_JSON = full_path(DATA_SET_TENDERS['tracker_dir'] + "/viewed-offers.json")

OBSERVED_FILE_PATH_IDS = full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/observed_ids.txt")
REPORTED_FILE_PATH_IDS = full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/reported_ids.txt")
VIEWED_FILE_PATH_IDS = full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids/viewed_ids.txt")

OBSERVED_BULLETIN_NUMBERS_PATH = full_path(DATA_SET_TENDERS['bzp_data_dir'] + "/bulletin_nums/observed_nums.txt")
REPORTED_BULLETIN_NUMBERS_PATH = full_path(DATA_SET_TENDERS['bzp_data_dir'] + "/bulletin_nums/reported_nums.txt")
VIEWED_BULLETIN_NUMBERS_PATH = full_path(DATA_SET_TENDERS['bzp_data_dir'] + "/bulletin_nums/viewed_nums.txt")


def parse_tracker_ids(tracker_data_file_path, out_valid_bulletin_numbers_file, out_ids_file):
    tender_ids = []
    with open(tracker_data_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        tenders_stats = json.load(file)
        for stat in tenders_stats:
            tender_ids.append(stat['what'])

        tender_ids = set(tender_ids)
        tender_ids = sorted(tender_ids)

        with open(out_valid_bulletin_numbers_file, 'w') as bulletin_nums_file:
            for tender_track_id in tender_ids:
                if tender_track_id.startswith('bzp-'):
                    id_number_suffix = tender_track_id[tender_track_id.rindex('-') + 1:].strip()
                    valid_bulletin_number = id_number_suffix + "-N-" + tender_track_id[4:8]
                    bulletin_nums_file.write("%s\n" % valid_bulletin_number)

        with open(out_ids_file, 'w') as ids_file:
            for tender_track_id in tender_ids:
                ids_file.write("%s\n" % tender_track_id)

    return


def load_bulletin_nums(bulletin_nums_file_path: str):
    bulletin_nums = []
    with open(bulletin_nums_file_path, 'r') as bulletin_nums_file:
        for line in bulletin_nums_file:
            bulletin_nums.append(line.strip())
    return bulletin_nums


os.makedirs(full_path(DATA_SET_TENDERS['tracker_dir'] + "/ids"), exist_ok=True)
os.makedirs(full_path(DATA_SET_TENDERS['bzp_data_dir'] + "/bulletin_nums"), exist_ok=True)

# parse_tracker_ids(TRACKER_OBSERVED_JSON, OBSERVED_BULLETIN_NUMBERS_PATH, OBSERVED_FILE_PATH_IDS)
# parse_tracker_ids(TRACKER_VIEWED_JSON, VIEWED_BULLETIN_NUMBERS_PATH, VIEWED_FILE_PATH_IDS)
# parse_tracker_ids(TRACKER_REPORTED_JSON, REPORTED_BULLETIN_NUMBERS_PATH, REPORTED_FILE_PATH_IDS)

bulletin_nums_of_tenders_to_find = {
    'observed': load_bulletin_nums(OBSERVED_BULLETIN_NUMBERS_PATH),
    'viewed': load_bulletin_nums(VIEWED_BULLETIN_NUMBERS_PATH),
    'reported': load_bulletin_nums(REPORTED_BULLETIN_NUMBERS_PATH)
}


def remove_existing_tenders_from_fetch_list(bulletin_nums_to_fetch: dict):
    jsons_dir_path = full_path(DATA_SET_TENDERS['bzp_data_jsons_dir'])
    for json_dir_name in os.listdir(jsons_dir_path):
        json_subdir_path = jsons_dir_path + "/" + json_dir_name
        for json_tender_file_name in os.listdir(json_subdir_path):
            tender_bulletin_num = json_tender_file_name.replace(".json", "")

            for key in bulletin_nums_to_fetch.keys():
                if tender_bulletin_num in bulletin_nums_to_fetch[key]:
                    bulletin_nums_to_fetch[key].remove(tender_bulletin_num)
    return


def tenders_to_fetch_count(bulletin_nums_to_fetch):
    tenders_left = sum(
        [len(bulletin_nums_to_fetch[key]) for key in bulletin_nums_to_fetch.keys()]
    )
    print('tenders left: {0}'.format(tenders_left))
    return tenders_left


remove_existing_tenders_from_fetch_list(bulletin_nums_of_tenders_to_find)


def print_tender_nums_to_fetch(bulletin_nums_to_fetch):
    for key in bulletin_nums_to_fetch.keys():
        for bulletin_num in bulletin_nums_to_fetch[key]:
            print(bulletin_num)


tenders_to_fetch_count = tenders_to_fetch_count(bulletin_nums_of_tenders_to_find)
if tenders_to_fetch_count > 0:
    print_tender_nums_to_fetch(bulletin_nums_of_tenders_to_find)
if tenders_to_fetch_count > 5:
    fetch_data_daily(bulletin_nums_of_tenders_to_find, full_path(DATA_SET_TENDERS['bzp_data_dir'] + "/jsons"))
