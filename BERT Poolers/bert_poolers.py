import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, BertModel,
    BertEmbeddings, BertEncoder, BertForSequenceClassification, BertPooler,
)
import numpy as np


class MeanMaxTokensBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states, *args, **kwargs):
        mean = hidden_states.mean(1)
        max, _ = hidden_states.max(1)
        pooled = torch.cat((mean, max), 1)
        pooled = self.linear(pooled)
        pooled = self.activation(pooled)
        return pooled

class MyBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = hidden_states.permute(0, 2, 1)
        batch_sz, token_sz, token_cnt = hidden_states.shape
        prob = softmax(hidden_states, dim=2)
        if self.training:
            #prob = prob.cpu().detach().numpy()#####0.02
            #indexes = np.apply_along_axis(pool_func, 2, prob, np.arange(token_cnt))#####0.89
            #output = [hidden_states[i, j, indexes[i, j]] for i in range(batch_sz) for j in range(token_sz)]#####0.15
            output = [hidden_states[i, j, k] for i in range(batch_sz) for (j, k) in enumerate(torch.multinomial(prob[i], 1))]
        else:
            output = [hidden_states[i, j].dot(prob[i, j]) for i in range(batch_sz) for j in range(token_sz)]#####1.3
        output = torch.stack(output).reshape(batch_sz, token_sz)
        output = self.linear(output)
        output = self.activation(output)
        return output


class MyBertConfig(BertConfig):
    def __init__(self, pooling_layer_type="CLS", **kwargs):
        super().__init__(**kwargs)
        self.pooling_layer_type = pooling_layer_type


class MyBertModel(BertModel):

    def __init__(self, config: MyBertConfig):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        if config.pooling_layer_type == "CLS":
            # See src/transformers/models/bert/modeling_bert.py#L610
            # at huggingface/transformers (9f43a425fe89cfc0e9b9aa7abd7dd44bcaccd79a)
            self.pooler = BertPooler(config)
        elif config.pooling_layer_type == "MEAN_MAX":
            self.pooler = MeanMaxTokensBertPooler(config)
        elif config.pooling_layer_type == "MINE":
            self.pooler = MyBertPooler(config)
        else:
            raise ValueError(f"Wrong pooling_layer_type: {config.pooling_layer_type}")

        self.init_weights()

    @property
    def pooling_layer_type(self):
        return self.config.pooling_layer_type


class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return super().forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states, return_dict
            )
