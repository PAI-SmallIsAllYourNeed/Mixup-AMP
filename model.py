import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertEncoder, BertPooler, BertLayerNorm

def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = (out.T*lam).T + (out[indices].T*(1-lam)).T
    target_shuffled_onehot = target_reweighted[indices]
    lam2 = lam.clone().detach()
    target_reweighted = (target_reweighted.T * lam2).T + (target_shuffled_onehot.T * (1 - lam2)).T
    #print(out,target_reweighted)
    return out, target_reweighted

class SoftEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        nn.Module.__init__(self)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))

    def forward(self, ids_or_probs, use_probs=False):
        if not use_probs:
            ids = ids_or_probs
            assert len(ids.shape) == 2
            probs = torch.zeros(
                ids.shape[0], ids.shape[1], self.num_embeddings,
                device=ids_or_probs.device).scatter_(2, ids.unsqueeze(2), 1.)
        else:
            probs = ids_or_probs

        embedding = probs.view(-1, self.num_embeddings).mm(self.weight). \
            view(probs.shape[0], probs.shape[1], self.embedding_dim)

        return embedding


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = SoftEmbedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = SoftEmbedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SoftEmbedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids_or_probs, token_type_ids=None,
                use_input_probs=False):
        seq_length = input_ids_or_probs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids_or_probs.device)
        position_ids = position_ids.unsqueeze(0). \
            expand(input_ids_or_probs.shape[:2])
        assert token_type_ids is not None
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = \
            self.word_embeddings(input_ids_or_probs, use_probs=use_input_probs)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = \
            words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(BertPreTrainedModel):
    def __init__(self, config,):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, input_ids_or_probs, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=False, use_input_probs=False):
        assert attention_mask is not None
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = \
            extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = \
            self.embeddings(input_ids_or_probs, token_type_ids, use_input_probs)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            head_mask=torch.Tensor([1] * self.config.num_attention_heads).long())
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

    def get_token_embeddings(self, input_ids_or_probs, token_type_ids=None, attention_mask=None,
                             use_input_probs=False):
        assert attention_mask is not None
        embedding_output = \
            self.embeddings(input_ids_or_probs, token_type_ids, use_input_probs)
        return embedding_output

    def get_sent_embedding(self, token_embeddings, attention_mask=None):
        assert attention_mask is not None
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = \
            extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.encoder(
            token_embeddings, extended_attention_mask,
            head_mask=torch.Tensor([1] * self.config.num_attention_heads).long())
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = kwargs['CLASS_SIZE']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, kwargs['CLASS_SIZE'])
        self.init_weights()

    def forward(self, input_ids_or_probs, token_type_ids=None,
                attention_mask=None, use_input_probs=False, target=None, mixup_hidden = False, layer_mix=None, lam=None):

        if mixup_hidden:
            if layer_mix == None:
                layer_mix = random.randint(0, 2)  # random mixup in different layers
            if lam==None:
                lam = np.random.beta(1, 1) # random lam if not set
            x = self.bert.get_token_embeddings(input_ids_or_probs, token_type_ids, attention_mask)
            if layer_mix==0:
                x, target = mixup_process(x, target, lam)
            x = self.bert.get_sent_embedding(x, attention_mask)
            if layer_mix==1:
                x, target = mixup_process(x, target, lam)
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits, target

        else:
            _, pooled_output = self.bert(
                input_ids_or_probs, token_type_ids, attention_mask, use_input_probs=use_input_probs)

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits





class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1
        if self.DROPOUT_PROB>0 and self.DROPOUT_PROB<1:
            self.dropout = nn.Dropout(self.DROPOUT_PROB)
        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
        self.conv = nn.ModuleList()
        for i in range(len(self.FILTERS)):
            self.conv.append(nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM))
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        self.init()
        # self.beta = torch.distributions.beta.Beta(torch.tensor([kwargs['ALPHA_A']]), torch.tensor([kwargs['ALPHA_B']]))

    def init(self):
        for name, weight  in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                    torch.nn.init.xavier_normal_(weight,gain=2)
            #todo fc初始化，实验
            # if 'fc' in name and 'weight' in name:
            #         torch.nn.init.xavier_normal_(weight,gain=2)

    def forward(self, x, target=None, mixup_hidden = False, layer_mix=None, lam=None):
        # target is one_hot vector
        x = x.long() # make sure the input type is long
        if mixup_hidden: # if we make mixup in hidden state
            if layer_mix == None:
                layer_mix = random.randint(0, 2)  # random mixup in different layers
            if lam==None:
                lam = np.random.beta(1, 1) # random lam if not set
            # layer#0
            x = self.embedding(x).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            if layer_mix == 0: # word
                x, target = mixup_process(x, target, lam)
            # layer#1
            conv_results = [
                F.max_pool1d(F.relu(self.conv[i](x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                    .view(-1, self.FILTER_NUM[i])
                for i in range(len(self.FILTERS))]
            x = torch.cat(conv_results, 1)
            if layer_mix == 1: # sentence
                x, target = mixup_process(x, target, lam)
            # layer#2
            if self.DROPOUT_PROB>0 and self.DROPOUT_PROB<1:
                x =self.dropout(x)
            x = self.fc(x)
            if layer_mix == 2:
                x, target = mixup_process(x, target, lam)
            return x, target

        else:

            x = self.embedding(x).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            conv_results = [
                F.max_pool1d(F.relu(self.conv[i](x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                    .view(-1, self.FILTER_NUM[i])
                for i in range(len(self.FILTERS))]
            x = torch.cat(conv_results, 1)
            if self.DROPOUT_PROB > 0 and self.DROPOUT_PROB < 1:
                x = self.dropout(x)
            x = self.fc(x)
            return x


class RNN(nn.Module):
    def __init__(self, **kwargs):
        super(RNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.HIDDEN_SIZE = kwargs["HIDDEN_SIZE"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        if self.DROPOUT_PROB<0 or self.DROPOUT_PROB>1:
            self.DROPOUT_PROB=0

        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.WV_MATRIX = kwargs["WV_MATRIX"]
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.lstm = nn.LSTM(self.WORD_DIM, self.HIDDEN_SIZE, 2,
                            bidirectional=True, batch_first=True, dropout=self.DROPOUT_PROB)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(self.HIDDEN_SIZE * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(self.HIDDEN_SIZE * 2, self.CLASS_SIZE)


    def forward(self, x, target=None, mixup_hidden = False, layer_mix=None, lam=None):
        x = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        if mixup_hidden:
            if layer_mix == None:
                layer_mix = random.randint(0, 2)  # random mixup in different layers
            if lam==None:
                lam = np.random.beta(1, 1) # random lam if not set
            # layer#0
            if layer_mix == 0: # word
                x, target = mixup_process(x, target, lam)
            H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

            M = self.tanh1(H)  # [128, 32, 256]
            # M = torch.tanh(torch.matmul(H, self.u))
            alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
            out = H * alpha  # [128, 32, 256]
            out = torch.sum(out, 1)  # [128, 256]
            if layer_mix == 1: # sentence
                out, target = mixup_process(out, target, lam)
            out = F.relu(out)
            out = self.fc1(out)
            return out, target
        else:
            H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

            M = self.tanh1(H)  # [128, 32, 256]
            # M = torch.tanh(torch.matmul(H, self.u))
            alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
            out = H * alpha  # [128, 32, 256]
            out = torch.sum(out, 1)  # [128, 256]
            out = F.relu(out)
            out = self.fc1(out)
        return out
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="test", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="MR", help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--mixup", default=0, type=int, help="0: no mixup, 1: mixup, 2: our mixup")
    parser.add_argument("--epoch", default=10, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--l2", default=0.004, type=float, help="l2")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--device", default="cuda:0", type=str, help="the device to be used")

    options = parser.parse_args()
    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": 123,
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": 111,
        "CLASS_SIZE": 3,
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "MIXUP":options.mixup,
        "L2":options.l2,
        "SEED":options.seed,
        "DEVICE": torch.device(options.device)
    }
    cnn = CNN(**params)
    cnn.init()
