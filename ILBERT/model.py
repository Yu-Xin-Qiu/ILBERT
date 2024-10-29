import torch
import torch.nn.functional as F
from torch import nn
from torch import nn
from transformers import RobertaConfig, RobertaModel







class CNN(nn.Module):
    """Network for fine-tuning on IL properties datasets"""
    def __init__(self,
                 dropout,
                 embed_size,
                 output_size=1,
                 num_filters=(100, 200, 200, 200, 200, 100, 100),
                 ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7),
                 IL_num_filters=(100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160),
                 IL_ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15)):
        super(CNN, self).__init__()
        
        self.num_filters = num_filters
        self.IL_num_filters = IL_num_filters
        
        self.IL_textcnn = nn.ModuleList([nn.Conv1d(in_channels=embed_size, out_channels=nf, kernel_size=ks)
                                        for nf, ks in zip(IL_num_filters, IL_ngram_filter_sizes)])
        self.output = nn.Linear(sum(IL_num_filters), embed_size)


    def forward(self, IL_src_nd):
        IL_encoded = IL_src_nd.permute(1,2,0)

        IL_textcnn_out = [F.relu(conv(IL_encoded)) for conv in self.IL_textcnn]
        IL_textcnn_out = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in IL_textcnn_out]  # Max pooling
        IL_textcnn_out = torch.cat(IL_textcnn_out, 1)  # Concatenate all the pooled features

        input_vecs = IL_textcnn_out
        out = self.output(input_vecs.float())

        return out




class ILBERT(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  
            hidden_size=d_model,  
            num_hidden_layers=nlayers,  
            num_attention_heads=nhead,  
            intermediate_size=d_hid, 
            hidden_dropout_prob=dropout,  
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input):
        x, _ = input
        attention_mask = (x != 0).long()
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  

        output = self.CNN(last_hidden_states)
        output = self.pred_head(output)

        return output




class ILBERT_T(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  
            hidden_size=d_model,  
            num_hidden_layers=nlayers,
            num_attention_heads=nhead, 
            intermediate_size=d_hid,  
            hidden_dropout_prob=dropout,  
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model+1, d_model//2),
            nn.Softplus(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, input):
        x, T = input
        attention_mask = (x != 0).long()

        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)

        output = self.CNN(last_hidden_states)
        T=T.view(-1, 1)

        output = self.pred_head(torch.cat((output, T.float()), dim=1))

        return output



class ILBERT_T_P(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  
            hidden_size=d_model,  
            num_hidden_layers=nlayers,  
            num_attention_heads=nhead,  
            intermediate_size=d_hid,  
            hidden_dropout_prob=dropout,  
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model+2, d_model//2),
            nn.Softplus(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, input):

        x, T,P = input
        attention_mask = (x != 0).long()

        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2) 

        output = self.CNN(last_hidden_states)

        T,P=T.view(-1, 1),P.view(-1, 1)

        output = self.pred_head(torch.cat((output, T.float(),P.float()), dim=1))

        return output
