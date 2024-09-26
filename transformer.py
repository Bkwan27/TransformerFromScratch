import torch
import torch.nn as nn
import torch.nn.functional as F
import random

max_len = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
eval_iterations = 200
n_iterations = 5000
eval = 100

def batch(data):
    ran_ind = random.choice(range(len(data) - max_len), batch_size)

    x = [data[i:i+max_len] for i in ran_ind]
    y = [data[i+1:i+1+max_len] for i in ran_ind]
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    return x, y

@torch.no_grad()
def evaluation(model, data):
    losses = torch.zeros(eval_iterations, dtype=torch.long)
    model.eval()
    for i in range(eval_iterations):
        b_data, targets = batch(data)
        logits = model(b_data)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        loss = F.cross_entropy(logits, targets)
        losses[i] = loss
    model.train()
    return losses.mean(dim=-1)


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



class SublayerConnection(nn.Module):
    def __init__(self, size, drop_prob):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.layernorm = LayerNorm(size)
    
    def forward(self, x, sublayer):
        # we have a residual connection here
        x = x + self.layernorm(self.dropout(sublayer(x)))
        return x


    
# class Decoder(nn.Module):
#     def __init__(self, layer, N):
#         self.layers = copy(layer, N)
#         self.LayerNorm = LayerNorm(layer.size)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return self.LayerNorm(x)

# class DecoderLayer(nn.Module):
#     def __init__(self, size, dropout):
#         self.attention1
#         self.ff
#         self.attention2
#         self.sublayer = copy(SublayerConnection(size, dropout), 3)

#     def forward(self, x):
#         x = self.sublayer[0](x, self.attention1)
#         x = self.sublayer[1](x, self.attention2)
#         x = self.sublayer[2](x, self.ff)
#         return x
# ### TODO: add the masks for the various transformer layers

# class 


# class Tokenizer(nn.Module):
#     def __init__(self):
#         super(Tokenizer, self).__init__()


class AttentionHead(nn.Module):
    def __init__(self, features, size, dropout=None):
        super(AttentionHead, self).__init__()
        self.query = nn.Linear(features, size)
        self.key = nn.Linear(features, size)
        self.value = nn.Linear(features, size)
        self.dropout = nn.Dropout(dropout)
        # self.size
    
    def attention(self, Q, K, V):
        # scaled dot product attention
        ei = torch.matmul(Q, K.transpose(-2, -1))
        ei /= torch.sqrt(Q.size)
        ei = F.softmax(ei)
        if self.dropout:
            ei = self.dropout(ei)
        attn = torch.matmul(ei, V)
        return attn

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        x = self.attention(Q, K, V)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, dim, h, dropout=0.0):
        self.heads = copy(AttentionHead(embed_size, dim), h)
        self.proj == nn.Linear(dim, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))
        
    
class PositionWiseFF(nn.Module):
    def __init__(self, embedding_size, dropout=None):
        super(PositionWiseFF, self).__init__()
        self.linear1 = nn.Linear(embedding_size, 4*embedding_size)
        self.linear2 = nn.Linear(4*embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, size, h, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(size, size, h, dropout)
        self.ff = PositionWiseFF(size, dropout)
        self.sublayer = copy(SublayerConnection(size, dropout), 2)
    
    def forward(self, x):
        x = self.sublayer[0](x, self.attention)
        x = self.sublayer[1](x, self.ff)
        return x

class Encoder(nn.Module):
    def __init__(self, embedding_size, N):
        super(Encoder, self).__init__()
        layer = EncoderLayer(embedding_size, 4, 0.5)
        self.layers = copy(layer, N)
        self.LayerNorm = LayerNorm(embedding_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.LayerNorm(x)
    
class Tokenizer(nn.Module):
    def __init__(self, embed_size, vocab):
        super(Tokenizer, self).__init__()
        self.embedding = nn.Embedding(vocab, embed_size)
    
    def forward(self, x):
        return self.embedding(x)
    
class Positional(nn.Module):
    def __init__(self, embed_size, dropout, max_len=32):
        super(Positional, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(max_len, embed_size)
        pos = torch.arange(0, max_len)
        div = 10000 ** torch.arange(0, embed_size, 2) / embed_size
        
        pos = pos/div
        self.pe[:, 0::2] = torch.sin(pos)
        self.pe[:, 1::2] = torch.cos(pos)
        self.pe = self.pe.unsqueeze(0)

    
    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)].requires_grad(False))


class Transformer(nn.Module):
    def __init__(self, input_dim, embedding_size):
        super(Transformer, self).__init__()
        # tokenizes sentences into embedding space
        self.tokenizer = Tokenizer(embedding_size, input_dim)
        # we first add positional to tokenized embedding
        self.positional = Positional(embedding_size, 0.5, 32)
        # throw it through multi headed attention
        self.encoder = Encoder(embedding_size, 4)
        # then throw it through a feed forward
        self.ff = nn.Linear(embedding_size, input_dim)
        self.LayerNorm = LayerNorm(embedding_size)
        # apply regularization
    
    def forward(self, x):
        embeddings = self.tokenizer(x)
        embeddings = self.positional(embeddings)
        embeddings = self.encoder(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.ff(embeddings)
        return embeddings

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            x = x[:, -max_len:]

            logits = self(x)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            pred = torch.multinomial(probs, 1)

            x = torch.cat((x, pred), dim=1)
        return x

 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
input_dim = len(chars)
embed_size = 64

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda str : [stoi[ch] for ch in str]
decode = lambda idx : ''.join([itos[i] for i in idx])

data = torch.tensor(encode(text), dtype = torch.long)
split = int(0.9*input_dim)
train_data = data[:split]
val_data = data[split:]

model = Transformer(input_dim, embed_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), 1e-3)

for i in range(n_iterations):

    if i % eval == 0 or i == n_iterations-1:
        print(f'step : {i}')
        print(f'training loss : {evaluation(train_data):.4f}')
        print(f'validation loss : {evaluation(val_data):.4f}')
    
    xb, yb = batch(train_data)

    logits = model(xb)

    loss = F.cross_entropy(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

    



        
