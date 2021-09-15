import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import pickle

from data_loader import load_dataset
from model import CNN, RNN, BertForSequenceClassification

KEYS = ['CLASSIFIER', 'ADV_TYPE', 'ADV_FLAG', "LAYER_MIX", 'SEED', 'CV', 'ALPHA']


def log_name(params):
    file_name = "train_log/" + params['DATASET'] + "/"
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    for key in KEYS:
        if key in params:
            file_name += key + "_" + str(params[key]) + "_"
    path = file_name + ".log"
    return path


## TODO 加cv，加数据集
def set_seed(seed=7):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    # torch.backends.cudnn.deterministic=True # cudnn


def train(model, train_set, test_set, params, logger):
    bce_loss = nn.BCELoss(reduction='none').to(params['DEVICE'])
    softmax = nn.Softmax(dim=1).to(params['DEVICE'])
    criterion = nn.CrossEntropyLoss().to(params['DEVICE'])
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=params["LEARNING_RATE"], weight_decay=params["L2"])
    train_loader = DataLoader(train_set, batch_size=params["BATCH_SIZE"], shuffle=True)
    max_test_acc = 0
    cur_batch = 0
    cur_epoch = 0
    train_loss = 0
    test_loss = 0
    go_flag = True

    gamma = params["GAMMA"]
    adv_type = params["ADV_TYPE"]
    adv_flag = params["ADV_FLAG"]
    loss_delta_recorder = []
    while go_flag:
        cur_epoch += 1
        for batch in train_loader:
            model.train()
            cur_batch += 1
            if cur_batch == params['TRAIN_BATCH']:
                go_flag = False
                break
            if params['CLASSIFIER'] != "BERT":
                batch_x, batch_y = batch
                batch_x = batch_x.to(params['DEVICE'])
                batch_y = batch_y.to(params['DEVICE'])
                one_hot_batch_y = torch.nn.functional.one_hot(batch_y, params["CLASS_SIZE"])
            else:
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                input_ids = input_ids.to(params['DEVICE'])
                input_mask = input_mask.to(params['DEVICE'])
                segment_ids = segment_ids.to(params['DEVICE'])
                batch_y = label_ids.to(params['DEVICE'])
                one_hot_batch_y = torch.nn.functional.one_hot(batch_y, params["CLASS_SIZE"])

            # argmax lam loss
            lam = np.random.beta(params['ALPHA'], params['ALPHA'], one_hot_batch_y.size()[0])
            # lam = np.fmax(lam,1-lam)
            lam = torch.FloatTensor(lam).to(params['DEVICE'])
            if adv_type == 0:
                if params['CLASSIFIER'] != "BERT":
                    pred = model(batch_x)
                else:
                    pred = model(input_ids, segment_ids, input_mask)
                pred = softmax(pred)
                loss = criterion(pred, batch_y)
            if adv_type == 1:
                lam.requires_grad_()
                if params['CLASSIFIER'] != "BERT":
                    mixed_x, mixed_y = model(batch_x, target=one_hot_batch_y, mixup_hidden=params['MIX_HIDDEN'],
                                             layer_mix=params['LAYER_MIX'], lam=lam)
                else:
                    mixed_x, mixed_y = model(input_ids, segment_ids, input_mask, target=one_hot_batch_y,
                                             mixup_hidden=True,
                                             layer_mix=params['LAYER_MIX'], lam=lam)
                pred = softmax(mixed_x)
                loss = bce_loss(pred, mixed_y).sum(dim=1)

                if adv_flag:
                    # get gradient for lambda
                    torch.mean(loss).backward(retain_graph=True)
                    loss_pre = loss
                    lgrad = lam.grad
                    lam = lam.clone().detach() - gamma * lgrad
                    # renew the interpolated input and label
                    if params['CLASSIFIER'] != "BERT":
                        mixed_x_, mixed_y_ = model(batch_x, target=one_hot_batch_y, mixup_hidden=params['MIX_HIDDEN'],
                                                   layer_mix=params['LAYER_MIX'], lam=lam)
                    else:
                        mixed_x_, mixed_y_ = model(input_ids, segment_ids, input_mask, target=one_hot_batch_y,
                                                   mixup_hidden=True,
                                                   layer_mix=params['LAYER_MIX'], lam=lam)
                    pred = softmax(mixed_x_)
                    loss = bce_loss(pred, mixed_y_)
                    loss_cur = loss.sum(dim=1)
                    # get the loss delta, record the position that loss delta greater than zero
                    loss_delta = loss_cur - loss_pre
                    mask_ = (loss_delta > 0).nonzero()
                    # record all the loss delta that greater than zero, and get the mean and std
                    positive_loss_delta = loss_delta.index_select(0, mask_.view(-1))
                    loss_delta_recorder.append(positive_loss_delta)
                    cur_delta_std = torch.cat(loss_delta_recorder[-params['MOVING_AVG']:]).std()
                    cur_delta_mean = torch.cat(loss_delta_recorder[-params['MOVING_AVG']:]).mean()
                    # generate a masker to mask the loss delta >0
                    positive_mask = (loss_delta > cur_delta_mean).float()
                    # to re-weight the samples that loss delta is greater than zero, the weight is calculated by the normalized  the loss delta
                    weight = torch.ones_like(loss_pre) + (loss_delta - cur_delta_mean) / cur_delta_std * positive_mask
                    weight = weight.clone().detach()
                    mask_ = (loss_delta > 0).float()
                    # calculate the final loss
                    # if the loss delta is greater than zero, use the loss after adv
                    # if the loss delta is lower than zero, use the loss before adv
                    # re-weight the loss by the normalized loss delta
                    loss = weight * (loss_cur * mask_ + loss_pre * (1 - mask_))
                    # print("{:.4f}-{:.4f} rate: {:.2f}%".format(loss_pre.item(),loss_cur.item(),100*count/cur_batch))

                loss = torch.mean(loss)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()
            model.eval()
            test_acc, test_loss_ = test(model, test_set, params)
            test_loss += test_loss_

            if test_acc > max_test_acc:
                max_test_acc = test_acc
            logger.info(
                "{} batch {:6d}\ttrain_loss {:.4f}\ttest_loss {:.4f}\ttest_acc {:.4f}\tmax_acc {:.4f}".format(
                    params['DATASET'], cur_batch, train_loss / cur_batch, test_loss / cur_batch, test_acc,
                    max_test_acc))
    logger.info("max test acc: {:.4f}".format(max_test_acc))


def test(model, data_set, params):
    data_loader = DataLoader(data_set, batch_size=params['BATCH_SIZE'], shuffle=False, drop_last=False)
    y_list = []
    pred_list = []
    test_loss = 0
    criterion = nn.CrossEntropyLoss().to(params['DEVICE'])
    batch_count = 0
    for batch in data_loader:
        if params['CLASSIFIER'] != "BERT":
            batch_x, batch_y = batch
            batch_x = batch_x.to(params['DEVICE'])
            batch_y = batch_y.to("cpu")

        else:
            input_ids, input_mask, segment_ids, label_ids, _ = batch
            input_ids = input_ids.to(params['DEVICE'])
            input_mask = input_mask.to(params['DEVICE'])
            segment_ids = segment_ids.to(params['DEVICE'])
            batch_y = label_ids.to("cpu")
        y_list.append(batch_y)
        if params['CLASSIFIER'] != "BERT":
            pred = model(batch_x, mixup_hidden=False)
        else:
            pred = model(input_ids, segment_ids, input_mask, mixup_hidden=False)
        loss = criterion(pred.to('cpu'), batch_y)
        test_loss += loss.item()
        pred = torch.argmax(pred, axis=1)
        pred_list.append(pred)
        batch_count += 1
    y_list = torch.cat(y_list, dim=0)
    pred_list = torch.cat(pred_list, dim=0)
    y_list = y_list.data.numpy()
    pred_list = pred_list.cpu().data.numpy()
    acc = sum([1 if p == y else 0 for p, y in zip(pred_list, y_list)]) / len(pred_list)
    return acc, test_loss / batch_count


def load_glove_txt(file_path="glove.6B.300d.txt"):
    results = {}
    num_file = sum([1 for i in open(file_path, "r", encoding='utf8')])
    with open(file_path, 'r', encoding='utf8') as infile:
        for line in tqdm.tqdm(infile, total=num_file):
            data = line.strip().split(' ')
            word = data[0]
            vec = np.array(data[1:], dtype=np.float32)
            results[word] = vec
    return results


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="non-static",
                        help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--seed", default=123, type=int, help="seed")
    parser.add_argument("--mixup", default=2, type=int, help="0: no mixup, 1: mixup, 2: our mixup")
    parser.add_argument("--cv", default=0, type=int, help="cv: 0-9  |  none")
    parser.add_argument("--device", default="cpu", type=str, help="the device to be used")
    parser.add_argument("--alpha", default="1", type=float, help="the alpha")
    parser.add_argument("--norm_limit", default=10, type=float, help="the norm limit")
    parser.add_argument("--moving_avg", default=5, type=int, help="the norm limit")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size")

    parser.add_argument("--dropout", default="-1", type=float, help="dropout ratio, between 0 and 1.")
    parser.add_argument("--train_batch", default=8000, type=int, help="number of max batch")
    parser.add_argument("--l2", default=0, type=float, help="l2")
    parser.add_argument("--mix_hidden", default=True, action='store_true', help="whether mixup hidden statues or not")
    parser.add_argument("--layer_mix", default=0, type=int, help="the layer to perform mixup， 1 word， 2 sentence")

    parser.add_argument("--dataset", default="TREC", help="available datasets: TREC ,SST1, SST2, SUBJ, MR")
    parser.add_argument("--adv_flag", default=False, action='store_true', help="using adv or not")
    parser.add_argument("--adv_type", default=1, type=int, help="0: no mixup or 1:mixup")
    parser.add_argument("--scale_rate", default=1., type=float, help="scale rate")
    parser.add_argument("--gamma", default=0.002, type=float, help="gamma")
    parser.add_argument("--max_sent_len", default=-1, type=int, help="max_length")
    parser.add_argument("--classifier", default="CNN", type=str, help="CNN,RNN,BERT")

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    options = parser.parse_args()
    # if options.classifier != "BERT":
    #     if options.dataset in ['SUBJ', 'MR']:
    #         filename = "{}-{}.pkl".format(options.dataset, options.cv)
    #     else:
    #         filename = "{}-{}.pkl".format(options.dataset, 0)
    #     with open(filename, 'rb') as f:
    #         train_set, test_set, wv_matrix, params_tmp = pickle.load(f)
    # else:
    #     if options.dataset in ['SUBJ', 'MR']:
    #         filename = "{}-{}-{}.pkl".format(options.classifier, options.dataset, options.cv)
    #     else:
    #         filename = "{}-{}-{}.pkl".format(options.classifier, options.dataset, 0)
    #     with open(filename, 'rb') as f:
    #         train_set, test_set, params_tmp = pickle.load(f)

    set_seed(options.seed)
    word_vectors = None
    if options.max_sent_len > 0:
        options.MAX_SENT_LEN = options.max_sent_len
    train_set, test_set, data = load_dataset(options)
    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "TRAIN_BATCH": options.train_batch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": options.MAX_SENT_LEN,
        "BATCH_SIZE": options.batch_size,
        "WORD_DIM": 300,
        "HIDDEN_SIZE": 512,
        "VOCAB_SIZE": options.VOCAB_SIZE,
        "CLASS_SIZE": options.CLASS_SIZE,
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": options.dropout,
        "NORM_LIMIT": options.norm_limit,
        "MIXUP": options.mixup,
        "MIX_HIDDEN": options.mix_hidden,
        "LAYER_MIX": options.layer_mix,
        "CV": options.cv,
        "L2": options.l2,
        "CLASSIFIER": options.classifier,
        "ALPHA": options.alpha,
        "SEED": options.seed,
        "ADV_TYPE": options.adv_type,
        "ADV_FLAG": options.adv_flag,
        "GAMMA": options.gamma,
        "SCALE_RATE": options.scale_rate,
        "DEVICE": torch.device(options.device),
        "MOVING_AVG": options.moving_avg
    }
    handler = logging.FileHandler(log_name(params))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("=" * 20 + "INFORMATION" + "=" * 20)
    for key in params:
        logger.info([key, params[key]])
    logger.info("=" * 20 + "INFORMATION" + "=" * 20)

    if params["MODEL"] == "non-static" and params['CLASSIFIER'] != 'BERT':
        # load word2vec
        logger.info("loading Glove...")
        if word_vectors == None:
            word_vectors = load_glove_txt(file_path="glove.840B.300d.txt")
        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors:
                wv_matrix.append(word_vectors[word])
            else:
                # print(word)
                wv_matrix.append(np.random.uniform(-0.01, 0.01, params['WORD_DIM']).astype("float32"))
        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, params['WORD_DIM']).astype("float32"))
        wv_matrix.append(np.zeros(params['WORD_DIM']).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix
    if params['CLASSIFIER'] == "CNN":
        model = CNN(**params).to(params['DEVICE'])
    elif params['CLASSIFIER'] == "RNN":
        model = RNN(**params).to(params['DEVICE'])
    elif params['CLASSIFIER'] == "BERT":
        model = BertForSequenceClassification.from_pretrained('./bert-base-uncased', **params).to(params['DEVICE'])
    if options.mode == "train":
        logger.info("=" * 20 + "TRAINING STARTED" + "=" * 20)
        train(model, train_set, test_set, params, logger)
        logger.info("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    logger.removeHandler(handler)
    logger.removeHandler(console)


if __name__ == "__main__":
    main()
