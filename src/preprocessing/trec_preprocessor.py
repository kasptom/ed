import numpy as np

from src.configuration import get_vector_labels_file_name, DATA_SET_TREC, \
    get_batch_file_name_for_dataset, get_vector_words_directory_for_dataset
from src.preprocessing.document_as_w2v_groups import document_to_batch
from src.preprocessing.w2v_loader import load_google_w2v_model
from src.utils.get_file import full_path, create_file_and_folders_if_not_exist

LABELS = {
    'DESC': [],
    'HUM': [],
    'ENTY': [],
    'NUM': [],
    'LOC': []
}
CORPUS_FILE_NAME = full_path("data/trec/trec.corp")

# see http://cogcomp.org/Data/QA/QC/
TREC_TEXT_FILE_PATH = full_path("data/trec/trec.txt")
WORD_VECTORS_DIRECTORY = get_vector_words_directory_for_dataset('trec', DATA_SET_TREC)
LABELS_FILE_PATH = get_vector_labels_file_name('trec')


def trec_preprocess():
    create_corpus_file()


def create_corpus_file():
    model = load_google_w2v_model()

    with open(TREC_TEXT_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as file:
        iter_file = iter(file)
        for line in iter_file:
            label = line[:str(line).find(":")]
            if label in LABELS:
                line = line[str(line).find(" ") + 1:]
                LABELS[label].append(line)

    labels = []
    create_file_and_folders_if_not_exist(WORD_VECTORS_DIRECTORY)

    keys = list(LABELS.keys())
    counter = 0
    sublist_counter = 0
    document_available = True

    with open(CORPUS_FILE_NAME, 'w+') as corpus_file:
        while document_available:
            document_available = False
            for key in keys:
                if sublist_counter < len(LABELS[key]):
                    document_available = True

                    document = LABELS[key][sublist_counter]

                    corpus_file.write(document)

                    batch_file_name = get_batch_file_name_for_dataset(counter, DATA_SET_TREC)
                    create_file_and_folders_if_not_exist(batch_file_name)

                    word_vector = document_to_batch(document, model, DATA_SET_TREC['max_time_steps'])

                    np.save(batch_file_name, word_vector)
                    counter += 1

                    labels.append(_generate_label_vector(keys.index(key)))
            sublist_counter += 1

        create_file_and_folders_if_not_exist(LABELS_FILE_PATH)
        np.save(LABELS_FILE_PATH, labels)


def _generate_label_vector(label_idx: int) -> np.array:
    label_vec = np.zeros(len(LABELS))
    label_vec[label_idx] = 1
    return label_vec


trec_preprocess()
