STOP_LIST = set('for a of the and to in'.split())


class MyCorpus(object):
    def __init__(self, file_name, dictionary):
        self.file_name = file_name
        self.dictionary = dictionary

    def __iter__(self):
        for line in open(self.file_name, encoding='utf-8', errors='ignore'):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(
                [word for word in line.lower().split() if word not in STOP_LIST]
            )
