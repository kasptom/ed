import time

from src.configuration import get_summaries_file_name, DATA_SET, configuration_string
from src.utils.get_file import create_file_and_folders_if_not_exist


def log_summary(score: float, test_accuracy_percent: float, time_elapsed_sec: float):
    summaries_file_name = get_summaries_file_name(DATA_SET['label'])
    create_file_and_folders_if_not_exist(summaries_file_name)

    with open(summaries_file_name, encoding='utf-8', errors='ignore', mode="a+") as summary_file:
        summary_file.write("---------------------------------------------------\n")
        summary_file.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        summary_file.write(configuration_string() + "\n")
        summary_file.write("score: %f , test accuracy: %f%%, time elapsed: %f\n" %
                           (score,
                            test_accuracy_percent,
                            time_elapsed_sec))
        summary_file.write("---------------------------------------------------\n")
