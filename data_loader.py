import csv
import random
import re
import sys
import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers.tokenization_bert import BertTokenizer


def load_glove_txt(file_path="glove.840B.300d.txt"):
    results = {}
    num_file = sum([1 for i in open(file_path, "r", encoding='utf8')])
    with open(file_path, 'r', encoding='utf8') as infile:
        for line in tqdm.tqdm(infile, total=num_file):
            data = line.strip().split(' ')
            word = data[0]
            results[word] = 1
    return results


def clean_str(string):
    # string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("\'s", " \'s", string)
    string = re.sub("\'ve", " \'ve", string)
    string = re.sub("n\'t", " n\'t", string)
    string = re.sub("\'re", " \'re", string)
    string = re.sub("\'d", " \'d", string)
    string = re.sub("\'ll", " \'ll", string)
    string = re.sub('"', " ", string)
    string = re.sub("'", " ", string)
    string = re.sub("`", " ", string)
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"[\[\]<>/&#\^$%{}‘\.…*]", " ", string)
    # string = re.sub(",", " , ", string)
    # string = re.sub("!", " ! ", string)
    # string = re.sub("\(", " \( ", string)
    # string = re.sub("\)", " \) ", string)
    # string = re.sub("\?", " \? ", string)
    # string = re.sub("\\\?", "?", string)
    # string = re.sub("\s{2,}", " ", string)
    # string = re.sub("-", ' ', string)
    return string.strip().split()


def shuffle_data(x, y):
    idx = list(range(len(x)))
    np.random.shuffle(idx)
    new_x = []
    new_y = []
    for id_ in idx:
        new_x.append(x[id_])
        new_y.append(y[id_])
    return new_x, new_y


def read_TREC(cv=None, scale_rate=1):
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/TREC/" + mode + ".tsv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                x.append(clean_str(line[0]))
                y.append(line[1])

        if mode == "train":
            label2data = {}
            for x_, y_ in zip(x, y):
                if y_ not in label2data:
                    label2data[y_] = [x_]
                else:
                    label2data[y_].append(x_)
            new_train_x = []
            new_train_y = []
            for y_ in label2data.keys():
                train_idx = max(int(len(label2data[y_]) * scale_rate), 1)
                for x_ in label2data[y_][:train_idx]:
                    new_train_x.append(x_)
                    new_train_y.append(y_)
            x, y = shuffle_data(new_train_x, new_train_y)

            data["train_x"], data["train_y"] = x, y

        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")

    return data


def read_SST1(cv=None, scale_rate=1):
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/SST1/" + mode + ".tsv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                y.append(line[1])
                x.append(clean_str(line[0]))
                # x.append(line[0])
        if mode == "train":
            with open("data/SST1/stsa.fine.phrases.train", "r", encoding="utf-8", errors='ignore') as f:
                for line in f:
                    y.append(line[0])
                    x.append(clean_str(line[2:]))
            label2data = {}
            for x_, y_ in zip(x, y):
                if y_ not in label2data:
                    label2data[y_] = [x_]
                else:
                    label2data[y_].append(x_)
            new_train_x = []
            new_train_y = []
            for y_ in label2data.keys():
                train_idx = max(int(len(label2data[y_]) * scale_rate), 1)
                for x_ in label2data[y_][:train_idx]:
                    new_train_x.append(x_)
                    new_train_y.append(y_)

            x, y = shuffle_data(new_train_x, new_train_y)
            data["train_x"], data["train_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")
    return data


def read_SST2(cv=None, scale_rate=1):
    data = {}

    def read(mode):
        x, y = [], []
        with open("data/SST2/" + mode + ".tsv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                y.append(line[1])
                x.append(clean_str(line[0]))
                # x.append(line[0])
        if mode == "train":
            with open("data/SST2/stsa.binary.phrases.train", "r", encoding="utf-8", errors='ignore') as f:
                for line in f:
                    y.append(line[0])
                    x.append(clean_str(line[2:]))
            label2data = {}
            for x_, y_ in zip(x, y):
                if y_ not in label2data:
                    label2data[y_] = [x_]
                else:
                    label2data[y_].append(x_)
            new_train_x = []
            new_train_y = []
            for y_ in label2data.keys():
                train_idx = max(int(len(label2data[y_]) * scale_rate), 1)
                for x_ in label2data[y_][:train_idx]:
                    new_train_x.append(x_)
                    new_train_y.append(y_)
            x, y = shuffle_data(new_train_x, new_train_y)

            data["train_x"], data["train_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")
    return data


def read_SUBJ(cv=0, scale_rate=1):
    data = {}
    x, y = [], []
    with open("data/SUBJ/subj.all", "r", encoding="utf-8", errors='ignore') as f:
        # reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in f:
            x.append(clean_str(line[2:]))
            # x.append(line[0])
            y.append(line[0])
    idx = list(range(len(x)))
    np.random.shuffle(idx)
    test_index = cv  # 0-9
    train_x = []
    train_y = []

    test_x = []
    test_y = []
    for i, id_ in enumerate(idx):
        index = i % 10
        if index == test_index:
            test_x.append(x[id_])
            test_y.append(y[id_])
        else:
            train_x.append(x[id_])
            train_y.append(y[id_])

    label2data = {}
    for x_, y_ in zip(train_x, train_y):
        if y_ not in label2data:
            label2data[y_] = [x_]
        else:
            label2data[y_].append(x_)
    new_train_x = []
    new_train_y = []
    for y_ in label2data.keys():
        train_idx = max(int(len(label2data[y_]) * scale_rate), 1)
        for x_ in label2data[y_][:train_idx]:
            new_train_x.append(x_)
            new_train_y.append(y_)
    train_x, train_y = shuffle_data(new_train_x, new_train_y)
    data["train_x"], data["train_y"] = train_x, train_y
    data["test_x"], data["test_y"] = test_x, test_y
    return data


def read_MR(cv=0, scale_rate=1):
    data = {}
    x, y = [], []
    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(clean_str(line))
            y.append(1)
    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(clean_str(line))
            y.append(0)

    idx = list(range(len(x)))
    np.random.shuffle(idx)
    test_index = cv  # 0-9
    # dev_index = (cv+1)%10
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i, id_ in enumerate(idx):
        index = i % 10
        if index == test_index:
            test_x.append(x[id_])
            test_y.append(y[id_])
        else:
            train_x.append(x[id_])
            train_y.append(y[id_])

    label2data = {}
    for x_, y_ in zip(train_x, train_y):
        if y_ not in label2data:
            label2data[y_] = [x_]
        else:
            label2data[y_].append(x_)
    new_train_x = []
    new_train_y = []
    for y_ in label2data.keys():
        train_idx = max(int(len(label2data[y_]) * scale_rate), 1)
        for x_ in label2data[y_][:train_idx]:
            new_train_x.append(x_)
            new_train_y.append(y_)

    train_x, train_y = shuffle_data(new_train_x, new_train_y)
    data["train_x"], data["train_y"] = train_x, train_y
    data["test_x"], data["test_y"] = test_x, test_y
    return data


def refind_sent(sent, g_dict):
    new_sent = []
    for word in sent:
        if word in g_dict:
            new_sent.append(word)
        elif '-' in word:
            for wd in word.split('-'):
                new_sent.append(wd)
        elif '\/' in word:
            for wd in word.split('\/'):
                new_sent.append(wd)
        elif word.lower() in g_dict:
            new_sent.append(word.lower())
        else:
            continue
    return new_sent


def preprocess_data(data, VOCAB_SIZE, MAX_SENT_LEN, dtype='train'):
    x = []
    for sent in data[dtype + "_x"]:
        sent_tmp = [data['word_to_idx']["<BOS>"]]
        for word in sent:
            if len(sent_tmp) < MAX_SENT_LEN - 1:
                sent_tmp.append(data['word_to_idx'][word])
        sent_tmp.append(data['word_to_idx']["<EOS>"])
        if len(sent_tmp) < MAX_SENT_LEN:
            sent_tmp += [VOCAB_SIZE + 1] * (MAX_SENT_LEN - len(sent_tmp))
        x.append(sent_tmp)
    y = [data["classes"].index(c) for c in data[dtype + "_y"]]
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


def load_dataset(options):
    mod = sys.modules[__name__]
    if options.classifier != 'BERT':
        data = getattr(mod, f"read_{options.dataset}")(cv=options.cv, scale_rate=options.scale_rate)
        g_dict = load_glove_txt()
        for i in range(len(data['train_x'])):
            data['train_x'][i] = refind_sent(data['train_x'][i], g_dict)
        for i in range(len(data['test_x'])):
            data['test_x'][i] = refind_sent(data['test_x'][i], g_dict)
        data["vocab"] = sorted(
            list(set([w for sent in data["train_x"] + data["test_x"] for w in sent] + ["<BOS>", "<EOS>"])))
        data["classes"] = sorted(list(set(data["train_y"])))
        data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
        data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
        options.VOCAB_SIZE = len(data["vocab"])
        if not hasattr(options, 'MAX_SENT_LEN'):
            options.MAX_SENT_LEN = max([len(sent) for sent in data["train_x"] + data["test_x"]])
        options.CLASS_SIZE = len(data["classes"])
        train_x, train_y = preprocess_data(data, options.VOCAB_SIZE, options.MAX_SENT_LEN, 'train')
        train_set = TensorDataset(train_x, train_y)
        test_x, test_y = preprocess_data(data, options.VOCAB_SIZE, options.MAX_SENT_LEN, 'test')
        test_set = TensorDataset(test_x, test_y)
        return train_set, test_set, data
    else:
        data = {}
        dset = getattr(mod, f"{options.dataset}_Processor")(cv=options.cv)
        train_examples = dset.train_examples
        test_examples = dset.test_examples
        data['tokenizer'] = BertTokenizer(vocab_file='./bert-base-uncased/vocab.txt'
                                          , do_basic_tokenize=True)
        data["classes"] = sorted(list(set([z.label for z in train_examples])))
        options.CLASS_SIZE = len(data["classes"])
        options.VOCAB_SIZE = len(data['tokenizer'].vocab)
        if not hasattr(options, 'MAX_SENT_LEN'):
            setattr(options, 'MAX_SENT_LEN',
                    max([len(example.text_a.split(' ')) for example in train_examples + test_examples]) + 2)
            # print("max",max([len(example.text_a.split(' ')) for example in train_examples + test_examples]))
        train_set = _make_data_loader(train_examples, data["classes"], data['tokenizer'], options.MAX_SENT_LEN)
        test_set = _make_data_loader(test_examples, data["classes"], data['tokenizer'], options.MAX_SENT_LEN)
        return train_set, test_set, data


def _make_data_loader(examples, label_list, tokenizer, MAX_SEQ_LENGTH):
    all_features = _convert_examples_to_features(
        examples=examples,
        label_list=label_list,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        output_mode='classification')

    all_input_ids = torch.tensor(
        [f.input_ids for f in all_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in all_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in all_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in all_features], dtype=torch.long)
    all_ids = torch.arange(len(examples))

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ids)
    return dataset


def _convert_examples_to_features(examples, label_list, max_seq_length,
                                  tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        # print(len(input_ids),len(input_mask),len(segment_ids),max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then each
    # token that's truncated likely contains more information than a longer
    # sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def csv_reader(filename):
    print('read file:', filename)
    f = open(filename, 'r', encoding='utf8')
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    return reader


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __getitem__(self, item):
        return [self.input_ids, self.input_mask,
                self.segment_ids, self.label_id][item]


class DatasetProcessor:
    def get_train_examples(self):
        raise NotImplementedError

    def get_dev_examples(self):
        raise NotImplementedError

    def get_test_examples(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError


class SST1_Processor(DatasetProcessor):
    """Processor for the SST-5 data set."""

    def __init__(self, cv=0):
        train_file = "./data/SST1/train.tsv"
        test_file = "./data/SST1/test.tsv"
        print("processing train_file{},test_file".format(train_file, test_file))
        self._train_set, self._test_set = csv_reader(train_file), csv_reader(test_file)
        self.train_examples, self.test_examples = self.get_train_examples(), self.get_test_examples()
        x, y = [], []
        with open("data/SST1/stsa.fine.phrases.train", "r", encoding="utf-8", errors='ignore') as f:
            for line in f:
                y.append(line[0])
                x.append(line[2:])
        self.train_examples_extra = self._create_examples(zip(x, y), "train")
        self.train_examples = self.train_examples + self.train_examples_extra

    def get_train_examples(self):
        """See base class."""
        examples = self._create_examples(self._train_set, "train")
        print('getting train examples,len = ', len(examples))
        return examples

    def get_test_examples(self):
        """See base class."""
        examples = self._create_examples(self._test_set, "test")
        print('getting test examples,len = ', len(examples))
        return examples

    def get_labels(self):
        """See base class."""
        label_set = set()
        for example in self.train_examples:
            label_set.add(example.label)
        return sorted(list(label_set))

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(
                guid=guid,
                text_a=data[0],
                label=data[1]
            ))
        # return examples
        return examples


class SST2_Processor(DatasetProcessor):
    """Processor for the SST-5 data set."""

    def __init__(self, cv=0):
        train_file = "./data/SST2/train.tsv"
        test_file = "./data/SST2/test.tsv"
        x, y = [], []
        with open("data/SST2/stsa.binary.phrases.train", "r", encoding="utf-8", errors='ignore') as f:
            for line in f:
                y.append(line[0])
                x.append(line[2:])
        self.train_examples_extra = self._create_examples(zip(x, y), "train")
        print("processing train_file{},test_file".format(train_file, test_file))
        self._train_set, self._test_set = csv_reader(train_file), csv_reader(test_file)

        self.train_examples, self.test_examples = self.get_train_examples(), self.get_test_examples()
        self.train_examples = self.train_examples + self.train_examples_extra

    def get_train_examples(self):
        """See base class."""
        examples = self._create_examples(self._train_set, "train")
        print('getting train examples,len = ', len(examples))
        return examples

    def get_test_examples(self):
        """See base class."""
        examples = self._create_examples(self._test_set, "test")
        print('getting test examples,len = ', len(examples))
        return examples

    def get_labels(self):
        """See base class."""
        label_set = set()
        for example in self.train_examples:
            label_set.add(example.label)
        return sorted(list(label_set))

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(
                guid=guid,
                text_a=data[0],
                label=data[1]
            ))
        # return examples
        return examples


class TREC_Processor(DatasetProcessor):
    """Processor for the SST-5 data set."""

    def __init__(self, cv=0):
        train_file = "./data/TREC/train.tsv"
        test_file = "./data/TREC/test.tsv"
        print("processing train_file{},test_file,{}".format(train_file, test_file))
        self._train_set, self._test_set = csv_reader(train_file), csv_reader(test_file)
        self.train_examples, self.test_examples = self.get_train_examples(), self.get_test_examples()

    def get_train_examples(self):
        """See base class."""
        examples = self._create_examples(self._train_set, "train")
        print('getting train examples,len = ', len(examples))
        return examples

    def get_test_examples(self):
        """See base class."""
        examples = self._create_examples(self._test_set, "test")
        print('getting test examples,len = ', len(examples))
        return examples

    def get_labels(self):
        """See base class."""
        label_set = set()
        for example in self.train_examples:
            label_set.add(example.label)
        return sorted(list(label_set))

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(
                guid=guid,
                text_a=data[0],
                label=data[1]
            ))
        # return examples
        return examples


class SUBJ_Processor(DatasetProcessor):
    """Processor for the SST-5 data set."""

    def __init__(self, cv):
        all_file = "./data/SUBJ/data_all.tsv"
        print("processing all_file{}".format(all_file))
        self._all_set = csv_reader(all_file)
        self.train_examples, self.test_examples = self.get_train_examples(cv=cv)

    def _read_examples(self):
        examples = self._create_examples(self._all_set, "all")
        return examples

    def get_train_examples(self, cv=0):
        """See base class."""
        examples = self._read_examples()
        idx = list(range(len(examples)))
        np.random.shuffle(idx)
        test_index = cv
        test_example = []
        train_example = []
        for i, id_ in enumerate(idx):
            index = i % 10
            if index == test_index:
                test_example.append(examples[id_])
            else:
                train_example.append(examples[id_])
        return train_example, test_example

    def get_labels(self):
        """See base class."""
        label_set = set()
        for example in self.train_examples:
            label_set.add(example.label)
        return sorted(list(label_set))

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(
                guid=guid,
                text_a=data[0],
                label=data[1]
            ))
        return examples
        # return shuffle_data(examples)


class MR_Processor(DatasetProcessor):
    """Processor for the SST-5 data set."""

    def __init__(self, cv=0):
        pos_file = "./data/MR/rt-polarity.pos"
        neg_file = "./data/MR/rt-polarity.neg"
        print("processing pos_file:{},neg_file:{}".format(pos_file, neg_file))
        self._pos_set, self._neg_set = csv_reader(pos_file), csv_reader(neg_file)
        self.train_examples, self.test_examples = self.get_train_examples(cv=cv)

    def _read_examples(self):
        pos_examples = self._create_examples(self._pos_set, "pos")
        neg_examples = self._create_examples(self._neg_set, "neg")
        examples = []
        for ex in pos_examples:
            examples.append(InputExample(
                guid=ex.guid,
                text_a=ex.text_a,
                label=1
            ))
        for ex in neg_examples:
            examples.append(InputExample(
                guid=ex.guid,
                text_a=ex.text_a,
                label=0
            ))
        return examples

    def get_train_examples(self, cv=0):
        """See base class."""
        examples = self._read_examples()
        idx = list(range(len(examples)))
        np.random.shuffle(idx)
        test_index = cv
        test_example = []
        train_example = []
        for i, id_ in enumerate(idx):
            index = i % 10
            if index == test_index:
                test_example.append(examples[id_])
            else:
                train_example.append(examples[id_])
        return train_example, test_example

    def get_labels(self):
        """See base class."""
        label_set = set()
        for example in self.train_examples:
            label_set.add(example.label)
        return sorted(list(label_set))

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(
                guid=guid,
                text_a=data[0],
            ))
        return examples


if __name__ == "__main__":
    processor = TREC_Processor(cv=2)
    print(processor.get_labels())
    train = processor.train_examples
    for x in train:
        print(x.text_a, x.label)
        break

    # class OPT:
    #     def __init__(self):
    #         self.dataset="SUBJ"
    #         self.cv = "0"
    #         self.scale_rate=1
    #         self.MAX_SENT_LEN=-1
    # opt = OPT()
    # dset = getattr(sys.modules[__name__],'load_dataset')(opt)
    # for x in dset[0]:
    #     print(x)
    #     break
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(dset[0], batch_size=50, shuffle=True)
