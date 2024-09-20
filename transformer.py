import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        # we have them as parameters since they are learned
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon
    
    def forward(self, x):
        ((x - torch.mean(x, dim=-1))/torch.sqrt(torch.std(x, dim=-1) + self.epsilon))* self.gamma + self.beta

def copy(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = copy(layer, N)
        self.LayerNorm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.LayerNorm(x)

class SublayerConnection(nn.Module):
    def __init__(self, size, drop_prob):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.layernorm = LayerNorm(size)
    
    def forward(self, x, sublayer):
        # we have a residual connection here
        x = x + self.layernorm(self.dropout(sublayer(x)))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, size, dropout)
        super(EncoderLayer, self).__init__()
        self.attention
        self.ff
        self.sublayer = copy(SublayerConnection(size, dropout), 2)
    
    def forward(self, x):
        x = self.sublayer[0](x, self.attention)
        x = self.sublayer[1](x, self.ff)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer, N):
        self.layers = copy(layer, N)
        self.LayerNorm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.LayerNorm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, dropout):
        self.attention1
        self.ff
        self.attention2
        self.sublayer = copy(SublayerConnection(size, dropout), 3)

    def forward(self, x):
        x = self.sublayer[0](x, self.attention1)
        x = self.sublayer[1](x, self.attention2)
        x = self.sublayer[2](x, self.ff)
        return x
### TODO: add the masks for the various transformer layers

class 


class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer, self).__init__()


class AttentionHead(nn.Module):
    def __init__(self, features, size, dropout):
        super(AttentionHead, self).__init__()
        self.query = nn.Linear(features, size)
        self.key = nn.Linear(features, size)
        self.value = nn.Linear(features, size)
        self.dropout = nn.Dropout(dropout)
        # self.size
    
    def attention(self, Q, K, V, dropout=None):
        # scaled dot product attention
        ei = torch.matmul(Q, K.transpose(-2, -1))
        ei /= torch.sqrt(Q.size)
        ei = F.softmax(ei)
        if self.dropout:
            ei = self.dropout(ei)
        attn = torch.matmul(ei, V)
        return attn

    def forward()

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # tokenizes sentences into embedding space
        self.tokenizer
        # we first add positional to tokenized embedding
        self.positional
        # throw it through multi headed attention
        self.attn_head
        # then throw it through a feed forward
        self.ff
        # apply regularization
        # dropout for the feed forward?
        self.dropout
        # layer norm for the attention layers?
        self.layernorm
