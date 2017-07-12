import os
import pickle

import numpy

from utilities.ResourceManager import ResourceManager


class WordVectorsManager(ResourceManager):
    def __init__(self, corpus=None, dim=None, omit_non_english=False):
        super().__init__()

        wv_directory = "..\\embeddings\\"

        self.omit_non_english = omit_non_english
        self.wv_filename = "{}.{}d.txt".format(corpus, str(dim))
        self.parsed_filename = "{}.{}d.pickle".format(corpus, str(dim))
        self.wv_file_path = "{}\\{}".format(wv_directory, self.wv_filename)

    def is_ascii(self, text):
        try:
            mynewstring = text.encode('ascii')
            return True
        except:
            return False

    def write(self):
        if os.path.exists(self.wv_file_path):
            print('Indexing file {} ...'.format(self.wv_filename))
            embeddings_dict = {}
            f = open(self.wv_file_path, "r", encoding="utf-8")
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                coefs = numpy.asarray(values[1:], dtype='float32')
                # if not self.is_ascii(word):
                #     print(word)

                # if word.lower() in {'<unk>', "<unknown>"}:
                #     print(word)
                #     print("UNKNOWN")
                #     print()

                if self.omit_non_english and not self.is_ascii(word):
                    continue

                embeddings_dict[word] = coefs
            f.close()
            print('Found %s word vectors.' % len(embeddings_dict))

            with open(os.path.join(os.path.dirname(__file__), self.parsed_filename), 'wb') as pickle_file:
                pickle.dump(embeddings_dict, pickle_file)

        else:
            print("{} not found!".format(self.wv_file_path))
            raise FileNotFoundError

    def read(self):
        if os.path.exists(os.path.join(os.path.dirname(__file__), self.parsed_filename)):
            with open(os.path.join(os.path.dirname(__file__), self.parsed_filename), 'rb') as f:
                return pickle.load(f)
        else:
            self.write()
            return self.read()
