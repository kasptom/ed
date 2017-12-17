WORD_NUMERIC_VECTOR_SIZE = 100
EPOCHS_NUMBER = 3
DROPOUT = 0.2
RECURRENT_DROPOUT = 0.2
BATCH_SIZE = 16
TEST_DATA_PERCENTAGE = 30


def print_configuration():
    print("----------------Configuration----------------")
    print(
        "vector size: %d, "
        "epochs number: %d, "
        "dropout: %d, "
        "recurrent dropout: %d, "
        "batch size: %d, "
        "test data percentage: %d" %
        (WORD_NUMERIC_VECTOR_SIZE, EPOCHS_NUMBER, DROPOUT, RECURRENT_DROPOUT, BATCH_SIZE, TEST_DATA_PERCENTAGE))
    print("---------------------------------------------")
