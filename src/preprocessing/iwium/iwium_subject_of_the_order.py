import json
import re
from unidecode import unidecode
from os import listdir

from src.configuration import DATA_SET_TENDERS, DATA_SET_TENDERS_SHORT, DATA_SET_TENDERS_LONG
from src.utils.get_file import full_path


def add_lines_to_corpus(json_dir_path, corpus_file_name):
    json_file_names = [f for f in listdir(json_dir_path)]
    for json_file_name in json_file_names:
        add_line_to_corpus(json_dir_path + '/' + json_file_name, corpus_file_name)


def add_lines_to_corpus_short(json_dir_path, corpus_file_name):
    json_file_names = [f for f in listdir(json_dir_path)]
    counter = 0
    for json_file_name in json_file_names:
        if counter < 450:
            add_line_to_corpus(json_dir_path + '/' + json_file_name, corpus_file_name)
        counter += 1


def add_line_to_corpus(json_file_path, corpus_file_name):
    with open(json_file_path, encoding='utf-8') as json_file:
        json_object = json.load(json_file)
        data = re.sub('\s+', ' ', json_object['okreslenie_przedmiotu'].strip())
        data = re.sub('[^\w\s]|_]', ' ', data)
        data = unidecode(data)

        words = data.split()
        words = [word for word in words if len(word) > 2]

        with open(corpus_file_name, mode='a+', errors='ignore') as corpus_file:
            corpus_file.write(' '.join(words) + '\n')
        # print(words)


json_dir_observed = full_path(DATA_SET_TENDERS['bzp_data_jsons_dir'] + '/observed_json/')
json_dir_viewed = full_path(DATA_SET_TENDERS['bzp_data_jsons_dir'] + '/viewed_json/')
json_dir_reported = full_path(DATA_SET_TENDERS['bzp_data_jsons_dir'] + '/reported_json/')

positive_corpus_path = full_path(DATA_SET_TENDERS['positive'])
negative_corpus_path = full_path(DATA_SET_TENDERS['negative'])

if DATA_SET_TENDERS == DATA_SET_TENDERS_SHORT:
    add_lines_to_corpus_short(json_dir_observed, positive_corpus_path)
    add_lines_to_corpus_short(json_dir_viewed, positive_corpus_path)
    add_lines_to_corpus_short(json_dir_reported, negative_corpus_path)
elif DATA_SET_TENDERS == DATA_SET_TENDERS_LONG:
    add_lines_to_corpus(json_dir_viewed, positive_corpus_path)
    add_lines_to_corpus(json_dir_observed, positive_corpus_path)
    add_lines_to_corpus(json_dir_reported, negative_corpus_path)
