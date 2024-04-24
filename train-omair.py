# %% [markdown]
# # Download and import libraries

# %%
from torchtext.data.utils import get_tokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import itertools
from torch.nn import Transformer
import tqdm.notebook
import warnings
warnings.simplefilter("ignore")
import torch.nn as nn
import torch.nn.functional as F
import math,copy,re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import gc
print(torch.__version__)

# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

set_seed(42)

# %% [markdown]
# ## Transformer Model (based on Attention is All you Need, Vaswani et. al.)

# %%
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)

    def forward(self, x):
        out = self.embed_layer(x)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        pe = torch.zeros((self.max_seq_len, self.embed_dim))

        for pos in range(self.max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000**(i/self.embed_dim)))
                pe[pos, i+1] = math.cos(pos / (10000**(i/self.embed_dim)))
        
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, inp):
        inp = inp*math.sqrt(self.embed_dim)
        seq_len = inp.size(1)
        inp = inp + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return inp

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim

        self.head_dim = int(self.embed_dim/self.n_heads)

        self.query_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.out = nn.Linear(embed_dim, embed_dim)
    

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_len = key.size(1)

        seq_len_query = query.size(1)

        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim)
        query = query.view(batch_size, seq_len_query, self.n_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim)

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)
        
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        k_adj = k.transpose(-1,-2)

        # prdt = torch.einsum("bhqd,bhdk->bhqk", q, k_adj)
        prdt = torch.matmul(q, k_adj)

        if mask is not None:
            prdt = prdt.masked_fill(mask==0, float("-1e20"))

        prdt = prdt/math.sqrt(self.embed_dim)
        prdt = F.softmax(prdt, dim=-1)

        # attention = torch.einsum("bhqk,bhkd->bhqd", prdt, v)
        attention = torch.matmul(prdt, v)

        concat = attention.transpose(1,2).contiguous().view(batch_size, seq_len_query, self.head_dim*self.n_heads)

        out = self.out(concat)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8, expansion_factor=4):
        super(TransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor

        self.multiheadattention = MultiHeadAttention(self.embed_dim, self.n_heads)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim*self.expansion_factor),
            nn.ReLU(),
            nn.Linear(self.embed_dim*self.expansion_factor, self.embed_dim)            
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, key, query, value, mask=None):
        attention_out = self.multiheadattention(key, query, value, mask)  
        attention_residual_out = attention_out + query
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) 

        feed_forward_out = self.feed_forward(norm1_out)
        feed_forward_residual_out = feed_forward_out + norm1_out 
        norm2_out = self.dropout2(self.norm2(feed_forward_residual_out)) 

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self, max_seq_len, vocab_size, embed_size=512, num_layers=6, n_heads=8, expansion_factor=4):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embeddings(vocab_size, embed_size)
        self.positional_embeddings = PositionalEmbedding(max_seq_len, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, n_heads, expansion_factor) for i in range(num_layers)
        ])

    def forward(self, x, mask=None):
        embed = self.embedding_layer(x)
        out = self.positional_embeddings(embed)
    
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8, expansion_factor=4):
        super(DecoderBlock, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor

        self.transformer_block = TransformerBlock(embed_dim, n_heads, expansion_factor)
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, key, value, x, tgt_mask, src_mask=None):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, query, value, src_mask)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, max_seq_len, target_vocab_size, embed_dim=512, num_layers=6, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()

        self.word_embedding = Embeddings(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)

    def forward(self, x, enc_out, tgt_mask, src_mask=None):
        embed = self.word_embedding(x)
        x = self.position_embedding(embed)
     
        for layer in self.layers:
            x = layer(enc_out, enc_out, x, tgt_mask, src_mask)
            
        logits = self.fc_out(x)

        return logits

class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, max_seq_length, num_layers=6, expansion_factor=4, n_heads=8, device='cpu'):
        super(Transformer, self).__init__()
 
        self.src_pad_idx = -1
        self.tgt_pad_idx = -1
        self.device = device

        self.encoder = TransformerEncoder(max_seq_length, 
                                          src_vocab_size, 
                                          embed_dim, 
                                          num_layers=num_layers, 
                                          expansion_factor=expansion_factor, 
                                          n_heads=n_heads)
        
        self.decoder = TransformerDecoder(max_seq_length, 
                                          target_vocab_size, 
                                          embed_dim, 
                                          num_layers=num_layers, 
                                          expansion_factor=expansion_factor, 
                                          n_heads=n_heads)
        
    
    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).bool()
        tgt_pad_mask = (tgt.cpu() != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2).bool()
        tgt_mask = tgt_mask & tgt_pad_mask
        return tgt_mask.to(self.device)   
    
    def make_pad_mask(self, inp, pad_idx):
        mask = (inp != pad_idx).unsqueeze(1).unsqueeze(2).bool()
        return mask.to(self.device)
    
    def forward(self, src, tgt):
        tgt_mask = self.make_tgt_mask(tgt)
        src_mask = self.make_pad_mask(src, self.src_pad_idx)
        enc_out = self.encoder(src)
        outputs = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return outputs

# %% [markdown]
# ### Loading Dataset

# %%
import random
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import vocab
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# install spacy datasets
# !python3 -m spacy download de_core_news_sm
# !python3 -m spacy download en_core_web_sm

iwslt_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")

train, test = iwslt_dataset['train'], iwslt_dataset['test']

# %%
train, test = iwslt_dataset['train'], iwslt_dataset['test']
# multi30k = load_dataset("bentrevett/multi30k")
# multi30k
# train, test = multi30k['train'], multi30k['test']

# %%
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# %%
ger_counter = Counter()
eng_counter = Counter()
for data in tqdm(train):
    ger_counter.update(tokenizer_ger(data['translation']['de'].lower()))
    eng_counter.update(tokenizer_eng(data['translation']['en'].lower()))

# %%
ger_vocab = vocab(ger_counter, min_freq=2, specials=("<unk>", "<pad>", "<sos>", "<eos>"))
eng_vocab = vocab(eng_counter, min_freq=2, specials=("<unk>", "<pad>", "<sos>", "<eos>"))
ger_vocab.set_default_index(ger_vocab["<unk>"])
eng_vocab.set_default_index(eng_vocab["<unk>"])
print(f"Size of German Vocab : {len(ger_vocab)}\n Size of English Vocab : {len(eng_vocab)}")

# %%
text_transform_eng = lambda x: [eng_vocab['<sos>']] + [eng_vocab[token.lower()] for token in tokenizer_eng(x)] + [eng_vocab['<eos>']]
text_transform_ger = lambda x: [ger_vocab['<sos>']] + [ger_vocab[token.lower()] for token in tokenizer_ger(x)] + [ger_vocab['<eos>']]

# %%
def collate_batch(batch):
    src_list, tgt_list = [], []
    for data in batch:
        src_list.append(torch.tensor(text_transform_ger(data['translation']['de'])))
        tgt_list.append(torch.tensor(text_transform_eng(data['translation']['en'])))

    src_list = pad_sequence(src_list, padding_value=ger_vocab['<pad>']).T
    tgt_list = pad_sequence(tgt_list, padding_value=eng_vocab['<pad>']).T
    
    inp = {
        "src": src_list,
        "tgt": tgt_list
    }

    return inp

# %% [markdown]
# ### Setting Training Parameters and DataLoader

# %%
num_epochs = 1
batch_size = 16
learning_rate = 1e-3
weight_decay = 0.001
writer = SummaryWriter(f"runs/loss")

train_dataloader = DataLoader(train, 
                              collate_fn=collate_batch,
                              shuffle=True,
                              batch_size=batch_size,
                              pin_memory=True)
test_dataloader = DataLoader(test, 
                              collate_fn=collate_batch,
                              shuffle=False,
                              batch_size=batch_size,
                              pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model = Transformer(embed_dim=512, 
                                src_vocab_size=len(ger_vocab), 
                                target_vocab_size=len(eng_vocab), 
                                max_seq_length=200, 
                                num_layers=6, 
                                expansion_factor=4, 
                                n_heads=8,
                                device=device)
transformer_model.src_pad_idx = ger_vocab['<pad>']
transformer_model.tgt_pad_idx = eng_vocab['<pad>']

# %%
total_steps = num_epochs*math.ceil(len(train)/batch_size)

optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                               max_lr=learning_rate,
                                               total_steps=total_steps,
                                               pct_start=0.33,
                                               div_factor=1e3,
                                               final_div_factor=1e2)
criterion = nn.CrossEntropyLoss(ignore_index=eng_vocab['<pad>'])

transformer_model = transformer_model.to(device)

load_model = False
if load_model:
    transformer_model.load_state_dict(torch.load("/model/my_checkpoint.pth.tar", map_location=device)['state_dict'])


# %% [markdown]
# ### Beam Search Code (Naive Implementation)

# %%
def translate_seq_beam_search(model, src, device, k=2, max_len=50):
    model.eval()

    src_mask = model.make_pad_mask(src, model.src_pad_idx)
    with torch.no_grad():
        enc_out = model.encoder(src, src_mask)

    # beam search

    candidates = [(torch.LongTensor([eng_vocab['<sos>']]), 0.0)]

    final_translations = []

    for a in range(max_len):

        input_batch = torch.concat([c[0].unsqueeze(0) for c in candidates], dim=0).to(device)

        if a>0:
            enc_out_repeat = enc_out.repeat(input_batch.shape[0], 1, 1)
        else:
            enc_out_repeat = enc_out

        
        with torch.no_grad():
            output = model.decoder(input_batch, enc_out_repeat, model.make_tgt_mask(input_batch), src_mask).detach().cpu()
        output[:, :, :2] = float("-1e20")
        output = output[:, -1, :]
        output = F.log_softmax(output, dim=-1)


        topk_output = torch.topk(output, k, dim=-1)
        topk_tokens = topk_output.indices
        topk_scores = topk_output.values
        

        new_seq = torch.concat([torch.concat([torch.vstack([c[0] for _ in range(k)]), topk_tokens[i].reshape(-1,1)], dim=-1) for i,c in enumerate(candidates)], dim=0)
        new_scores = torch.concat([c[1] + topk_scores[i] for i,c in enumerate(candidates)], dim=0)


        topk_new = torch.topk(new_scores, k=k).indices.tolist()

        new_candidates = []

        for i in range(k):
            if new_seq[topk_new[i]][-1] == eng_vocab["<eos>"] or a==max_len-1:
                final_translations.append((new_seq[topk_new[i]].tolist(), int(new_scores[topk_new[i]])))
            else:
                new_candidate = (new_seq[topk_new[i]], new_scores[topk_new[i]])
                new_candidates.append(new_candidate)

        
        if len(new_candidates) > 0:
            candidates = new_candidates
        else:
            break
    

    return final_translations

# %% [markdown]
# ### Greedy Sequence Generation

# %%
def translate_seq(model, src, device, max_len=50):
    model.eval()
    src_mask = model.make_pad_mask(src, model.src_pad_idx)
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
    tgt_indexes = [eng_vocab["<sos>"]]
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)
        with torch.no_grad():
            output = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)
        output[:, :, :2] = float("-1e20")  # cannot predict <unk>, <pad> token
        output = output[:, -1, :] # pick the last token
        output = F.softmax(output, dim=-1)
        pred_token = output.argmax(-1).item()
        tgt_indexes.append(pred_token)
        if pred_token == eng_vocab["<eos>"]:
            break
    return tgt_indexes

# %% [markdown]
# ### Helper Functions

# %%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

# %% [markdown]
# ## Start Training

# %%
step = 0
for epoch in range(1, num_epochs+1):
    
    print(f"[Epoch {epoch} / {num_epochs}]")
    
    checkpoint = {"state_dict": transformer_model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, "/model/my_checkpoint.pth.tar")
    
    loss_meter = AvgMeter()
    transformer_model.train()

    bar = tqdm(train_dataloader, total=math.ceil(len(train)/batch_size))

    for idx, data in enumerate(bar):
        
        german = data["src"].to(device)
        english = data["tgt"].to(device)

        count = german.shape[0]

        output = transformer_model(german, english[:,:-1])
        
        output = output.reshape(-1, output.shape[2])
        english = english[:, 1:]
        english = english.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, english)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1)

        optimizer.step()
        
        if scheduler:
            scheduler.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
        
        loss_meter.update(loss.item(), count)
        bar.set_postfix(loss=loss_meter.avg, lr=get_lr(optimizer), step=step)
    
    # Example Generation (Greedy Decode)
    ex = test[random.randint(0, len(test))]
    sentence = ex['translation']['de']
    src_indexes = torch.tensor(text_transform_ger(sentence)).unsqueeze(0).to(device)    
    translated_sentence_idx = translate_seq(transformer_model, src_indexes, device=device, max_len=50)
    translated_sentence = [eng_vocab.get_itos()[i] for i in translated_sentence_idx]
    print(f"\nExample sentence: \n {sentence}\n")
    print(f"Original Translation : \n {' '.join(translated_sentence[1:-1])}\n")
    print(f"Generated Translation : \n{ex['en']}\n")
    
    del src_indexes, ex, sentence, translated_sentence_idx, translated_sentence, checkpoint
    torch.cuda.empty_cache()
    _ = gc.collect()

# %% [markdown]
# ### Sample Beam Search Generation from Test Data

# %%
for n in range(5):
    print(f"Example {n+1}\n")
    ex = test[random.randint(0, len(test))]
    sentence = ex['translation']['de']
    src_indexes = torch.tensor(text_transform_ger(sentence)).unsqueeze(0).to(device)    
    k = 3
    translated_sentence_ids = translate_seq_beam_search(transformer_model, src_indexes, k=k, device=device, max_len=50)
    translated_sentence_ids = sorted(translated_sentence_ids, key= lambda x: x[1], reverse=True)
    translations = [[eng_vocab.get_itos()[i] for i in translated_sentence[0]] for translated_sentence in translated_sentence_ids]
    print(f"German : {ex['translation']['de']}")
    print(f"English : {ex['translation']['en']}\n")
    print(f"English Translations generated:\n")
    for i in range(k):
        for w in translations[i]:
            if w in ['<sos>', '<eos>', '<pad>', '<unk>']:
                continue
            print(w, end=" ")
        print()
    print("---------------------------------------------------------------------\n")

del src_indexes, ex, sentence, translated_sentence_ids, translations
torch.cuda.empty_cache()
_ = gc.collect()

# %% [markdown]
# ## Calculating Bleu Score

# %%
from torchtext.data.metrics import bleu_score

def calculate_bleu(data, model, device, max_len=50):
    tgts = []
    preds = []
    for datum in tqdm(data):
        src = datum['translation']["de"]
        tgt = datum['translation']["en"]
        src_idx = torch.tensor(text_transform_ger(src)).unsqueeze(0).to(device)
        pred_tgt = translate_seq(model, src_idx, device, max_len)
        pred_tgt = pred_tgt[1:-1]
        pred_sent = [eng_vocab.get_itos()[i] for i in pred_tgt]
        preds.append(pred_sent)
        tgts.append([tokenizer_eng(tgt.lower())])

    return bleu_score(preds, tgts) 

# %%
bleu = calculate_bleu(test, transformer_model, device)
print("BLEU Score Achieved :", bleu)