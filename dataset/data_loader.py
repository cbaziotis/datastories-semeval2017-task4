import glob
import os
import re

from utilities.generic import clean_text


class SemEvalDataLoader:
    # Validation set is used for tuning the parameters of a model
    # Test set is used for performance evaluation

    # Training set      --> to fit the parameters [i.e., weights]
    # Validation set    --> to tune the parameters [i.e., architecture]
    # Test set          --> to assess the performance [i.e., generalization and predictive power]

    def __init__(self, verbose=True):
        self.verbose = verbose

        self.SEPARATOR = "\t"
        _dirname = os.path.dirname(__file__)
        self.task_folders = {
            "A": os.path.join(_dirname, "Subtask_A/downloaded/"),
            "BD": os.path.join(_dirname, "Subtask_BD/downloaded/"),
            "CE": os.path.join(_dirname, "Subtask_CE/downloaded/"),
        }
        print()

    def parse_file(self, filename, with_topic=False):
        """
        Reads the text file and returns a dictionary in the form:
        tweet_id = (sentiment, text)
        :param with_topic:
        :param filename: the complete file name
        :return:
        """
        data = {}
        fname_print_friendly = filename.split("/")[-1].split("\\")[-1]

        if self.verbose:
            print("Parsing file:", fname_print_friendly, end=" ")
        for line_id, line in enumerate(
                open(filename, "r", encoding="utf-8").readlines()):

            try:
                columns = line.rstrip().split(self.SEPARATOR)
                tweet_id = columns[0]

                if with_topic:
                    topic = clean_text(columns[1])
                    if not isinstance(topic, str) or "None" in topic:
                        print(tweet_id, topic)
                    sentiment = columns[2]
                    text = clean_text(" ".join(columns[3:]))

                    if text != "Not Available":
                        data[tweet_id] = (sentiment, (topic, text))
                else:
                    sentiment = columns[1]
                    text = clean_text(" ".join(columns[2:]))

                    if text != "Not Available":
                        data[tweet_id] = (sentiment, text)
            except Exception as e:
                print("\nWrong format in line:{} in file:{}".format(
                    line_id, fname_print_friendly))
                raise Exception

        if self.verbose:
            print("done!")
        return data

    def get_silver(self, no_seeds=True):
        data = []
        fname = "silver_seeds_omitted.txt" if no_seeds else "silver.txt"
        # print(fname)
        if self.verbose:
            print("Parsing file:", fname, end=" ")
        _path = os.path.join(os.path.dirname(__file__), "Subtask_A/" + fname)
        with open(_path, "r", encoding="utf-8")as f:
            for line in f:
                data.append(line.rstrip().split(self.SEPARATOR))
        if self.verbose:
            print("done!")
        return data

    def get_gold(self, task):
        fname = "SemEval2017-task4-test.subtask-{}.english.txt".format(task)
        task_dir = "Subtask_{}/gold/".format(task)
        file = os.path.join(os.path.dirname(__file__), task_dir, fname)
        data = self.parse_file(file, with_topic=task != "A")
        if self.verbose:
            print("done!")
        return [v for k, v in sorted(data.items())]

    def get_data(self, task, years=None, datasets=None, only_semeval=True):
        """
        Get the data from the downloaded folder for a given set of parameters
        :param task: the SemEval Task for which to get the data
        :param years: a number or a tuple of (from,to)
        :param datasets: set with possible values {"train", "dev", "devtest", "test"}
        :return: a list of tuples (sentiment, text)
        """
        if only_semeval:
            files = glob.glob(self.task_folders[task] + "*{}.tsv".format(task))
        else:
            files = glob.glob(self.task_folders[task] + "*.tsv")
        data = {}

        if years is not None and not isinstance(years, tuple):
            years = (years, years)

        for file in files:
            year = int(re.findall("\d{4}", file)[-1])
            _type = re.findall("(?<=\d{4})\w+(?=\-)", file)[-1]

            if _type not in {"train", "dev", "devtest", "test"}:
                _type = "devtest"

            if years is not None and not years[0] <= year <= years[1]:
                continue
            if datasets is not None and _type not in datasets:
                continue

            dataset = self.parse_file(file, with_topic=task != "A")
            # data.append((year, tp, dataset))
            data.update(dataset)

        return [v for k, v in sorted(data.items())]

# SemEvalDataLoader().get_gold(task="A")
