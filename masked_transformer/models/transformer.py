import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedPositionalEncoding(nn.Module): 
    """ Positional encoding layer
    Parameters
    ----------
    dropout : float
    Dropout value.
    num_embeddings : int
    Number of embeddings to train.
    hidden_dim : int
    Embedding dimensionality
    """
    
    def __init__(self, num_embeddings, hidden_dim, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
    
    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.hidden_dim)
        x = x + embeddings
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, seq_len, input_size, nhead, n_layers, dim_feedforward, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_size, dim_feedforward)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_feedforward))
        self.pos_encoder = LearnedPositionalEncoding(seq_len+1, dim_feedforward, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(dim_feedforward,
                                                   nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, src, src_key_padding_mask=None):
        B = src.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        src = self.embedding(src)
        src = torch.cat((cls_tokens, src), dim=1)
        src = self.pos_encoder(src)
        src = self.tf(src.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        return src.permute(1, 0, 2)

class Decoder(nn.Module):
    def __init__(self, seq_len, output_size, nhead, n_layers, dim_feedforward, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Linear(output_size, dim_feedforward)
        self.pos_decoder = LearnedPositionalEncoding(seq_len, dim_feedforward, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(dim_feedforward, 
                                                   nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout)
        self.tf = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, tgt, memory, 
                    tgt_mask=None, 
                    memory_mask=None, 
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.pos_decoder(tgt)
        tgt = self.tf(tgt.permute(1, 0, 2), memory.permute(1, 0, 2), 
                    tgt_mask=tgt_mask, 
                    memory_mask=memory_mask, 
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
        return tgt.permute(1, 0, 2)   
   
class Transformer(nn.Module):
    def __init__(self, seq_len, input_sizes, nheads, n_layers, dims_feedforward, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(seq_len, input_sizes[0], nheads[0], n_layers[0], dims_feedforward[0], dropout=dropout)
        self.transition = nn.Linear(dims_feedforward[0], dims_feedforward[1])
        self.decoder = Decoder(seq_len, input_sizes[1], nheads[1], n_layers[1], dims_feedforward[1], dropout=dropout)
        self.generator = nn.Linear(dims_feedforward[1], input_sizes[1]+1)
        
    def forward(self, src, tgt,
                src_mask=None, tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        e_outputs = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        e_outputs = self.transition(e_outputs)
        d_output = self.decoder(tgt, e_outputs, 
                                tgt_mask=tgt_mask, 
                                memory_mask=memory_mask, 
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        output = self.generator(d_output)
        return output  
    
    def encode(self, src, src_key_padding_mask=None):
        e_outputs = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        e_outputs = self.transition(e_outputs) 
        return e_outputs

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, input_size, nheads, n_layers, dim_feedforward, n_labels=2, dropout=0.1, avg_pool=False):
        super(TransformerEncoder, self).__init__()
        self.encoder = Encoder(seq_len, input_size, nheads, n_layers, dim_feedforward, dropout=dropout)
        self.avg_pool = avg_pool
        if avg_pool:
            self.avg_pool_layer = AvgPoolSequence() 
        self.linear = nn.Linear(dim_feedforward, n_labels)
        self.drop = nn.Dropout(p=dropout)
       
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        if self.avg_pool:
            output = self.avg_pool_layer(output, src_key_padding_mask)
        else:
            output = output[:,0]
        output = self.drop(self.linear(output))
        return output  

class AvgPoolSequence(nn.Module):
    """ Custom Average Pooling Layer with mask array """
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        b,t,d = x.shape[:3]
        mask_ = mask.unsqueeze(-1).expand(x.size())
        x = x.permute(0, 2, 1)
        mask_ = mask_.permute(0, 2, 1)
        x=x.view(b,d,-1)
        mask = ~mask_
        x = torch.mul(x,mask)
        nelements = mask.sum(dim=-1)
        pooling = x.sum(dim=-1)/nelements
        return pooling


