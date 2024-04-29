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
import spacy
# import seaborn as sns
# import matplotlib.pyplot as plt

import random
import gc 
from torch.utils.tensorboard import SummaryWriter
import spacy
from torchtext.vocab import vocab
from tqdm import tqdm


print(torch.__version__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

set_seed(42)# %% [markdown]

# ## Transformer Model (based on Attention is All you Need, Vaswani et. al.)
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)

    def forward(self, x):
        # 3.5 In the embedding layers, we multiply those weights by √dmodel.
        # Scale the embeddings by the square root of the embedding dimension
        out = self.embed_layer(x) * math.sqrt(self.embed_dim)
        # out = self.embed_layer(x)
        return out


## Paper implementation (3.5)
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Create a new zeroed tensor for positional encoding with shape (max_seq_len, embed_dim)
        pe = torch.zeros((self.max_seq_len, self.embed_dim))

        # Calculate the positional encoding values as per the paper's formula
        for pos in range(self.max_seq_len):
            for i in range(0, self.embed_dim // 2):  # use embed_dim // 2 for correct pairing of sine and cosine
                # Sine for even indices (2i)
                pe[pos, 2 * i] = math.sin(pos / (10000 ** (2 * i / self.embed_dim)))
                # Cosine for odd indices (2i + 1)
                pe[pos, 2 * i + 1] = math.cos(pos / (10000 ** (2 * i / self.embed_dim)))

        pe = pe.unsqueeze(0)  # Add a new dimension at the 0th position

        # Register pe as a buffer that is not a parameter but should be part of the module's state
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Multiply the input by the square root of the embedding dimension
        x = x * math.sqrt(self.embed_dim)
        # Retrieve the sequence length from the input dimensions
        seq_len = x.size(1)
        # Add the positional encoding to the input embedding, ensuring no gradient is calculated for pe
        x = x + self.pe[:, :seq_len]
        return x

## Paper implementation (3.2.2)
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"

        self.query_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_matrix = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)

        # Project and split into heads
        query = self.query_matrix(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.key_matrix(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.value_matrix(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Calculate dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)

        # Apply attention to value vector
        context = torch.matmul(attention, value)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(context)
        return output

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


# Adding shared weights according to paper
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
        # Share weights between embedding and pre-softmax linear transformation
        self.fc_out.weight = self.word_embedding.embed_layer.weight

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
    



# install spacy datasets
# !python3 -m spacy download de_core_news_sm
# !python3 -m spacy download en_core_web_sm

iwslt_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")

train, test = iwslt_dataset['train'], iwslt_dataset['test']

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

ger_counter = Counter()
eng_counter = Counter()
for data in tqdm(train):
    ger_counter.update(tokenizer_ger(data['translation']['de'].lower()))
    eng_counter.update(tokenizer_eng(data['translation']['en'].lower()))

ger_vocab = vocab(ger_counter, min_freq=2, specials=("<unk>", "<pad>", "<sos>", "<eos>"))
eng_vocab = vocab(eng_counter, min_freq=2, specials=("<unk>", "<pad>", "<sos>", "<eos>"))
ger_vocab.set_default_index(ger_vocab["<unk>"])
eng_vocab.set_default_index(eng_vocab["<unk>"])
print(f"Size of German Vocab : {len(ger_vocab)}\n Size of English Vocab : {len(eng_vocab)}")


text_transform_eng = lambda x: [eng_vocab['<sos>']] + [eng_vocab[token.lower()] for token in tokenizer_eng(x)] + [eng_vocab['<eos>']]
text_transform_ger = lambda x: [ger_vocab['<sos>']] + [ger_vocab[token.lower()] for token in tokenizer_ger(x)] + [ger_vocab['<eos>']]

def collate_batch(batch):
    src_list, tgt_list = [], []
    for data in batch:
        src_list.append(torch.tensor(text_transform_eng(data['translation']['en'])))
        tgt_list.append(torch.tensor(text_transform_ger(data['translation']['de'])))

    src_list = pad_sequence(src_list, padding_value=eng_vocab['<pad>']).T
    tgt_list = pad_sequence(tgt_list, padding_value=ger_vocab['<pad>']).T
    
    inp = {
        "src": src_list,
        "tgt": tgt_list
    }

    return inp


# ### Setting Training Parameters and DataLoader
num_epochs = 20
batch_size = 32
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
                                src_vocab_size=len(eng_vocab), 
                                target_vocab_size=len(ger_vocab), 
                                max_seq_length=200, 
                                num_layers=6, 
                                expansion_factor=4, 
                                n_heads=8,
                                device=device)
transformer_model.src_pad_idx = eng_vocab['<pad>']
transformer_model.tgt_pad_idx = ger_vocab['<pad>']


total_steps = num_epochs*math.ceil(len(train)/batch_size)

# optimizer = torch.optim.Adam(transformer_model.parameters(), lr=learning_rate)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
#                                                max_lr=learning_rate,
#                                                total_steps=total_steps,
#                                                pct_start=0.33,
#                                                div_factor=1e3,
#                                                final_div_factor=1e2)


# this is bettter 
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=1e-4)
scheduler = False

                                               

criterion = nn.CrossEntropyLoss(ignore_index=ger_vocab['<pad>'])

transformer_model = transformer_model.to(device)

load_model = True
if load_model:
    # transformer_model.load_state_dict(torch.load("/pscratch/sd/j/josh-ee/tf/model_ckpt/og_epoch5_checkpoint.pth.tar", map_location=device)['state_dict'])

    checkpoint = torch.load("/pscratch/sd/j/josh-ee/tf/model_ckpt/no_scheduler_epoch36_checkpoint.pth.tar", map_location=device)

    # Load the model state dictionary
    transformer_model.load_state_dict(checkpoint['state_dict'])

    # Load the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Retrieve the epoch number if needed
    epoch = checkpoint['epoch']



# ### Beam Search Code (Naive Implementation)
def translate_seq_beam_search(model, src, device, k=2, max_len=50):
    model.eval()

    src_mask = model.make_pad_mask(src, model.src_pad_idx)
    with torch.no_grad():
        enc_out = model.encoder(src, src_mask)

    # beam search

    candidates = [(torch.LongTensor([ger_vocab['<sos>']]), 0.0)]

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
            if new_seq[topk_new[i]][-1] == ger_vocab["<eos>"] or a==max_len-1:
                final_translations.append((new_seq[topk_new[i]].tolist(), int(new_scores[topk_new[i]])))
            else:
                new_candidate = (new_seq[topk_new[i]], new_scores[topk_new[i]])
                new_candidates.append(new_candidate)

        
        if len(new_candidates) > 0:
            candidates = new_candidates
        else:
            break
    

    return final_translations


# ### Greedy Sequence Generation
def translate_seq(model, src, device, max_len=50):
    model.eval()
    src_mask = model.make_pad_mask(src, model.src_pad_idx)
    with torch.no_grad():
        enc_src = model.encoder(src, src_mask)
    tgt_indexes = [ger_vocab["<sos>"]]
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
        if pred_token == ger_vocab["<eos>"]:
            break
    return tgt_indexes


# ### Helper Functions

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


if load_model:
    start = epoch +1
    end = start + num_epochs + 20
    step = epoch

else:
    start = 1   
    end = num_epochs + 1

# ## Start Training
step = 0
for epoch in range(start, end):
    
    print(f"[Epoch {epoch} / {end}]")
    
    loss_meter = AvgMeter()
    transformer_model.train()

    bar = tqdm(train_dataloader, total=math.ceil(len(train)/batch_size))

    for idx, data in enumerate(bar):
        
        english = data["src"].to(device)
        german = data["tgt"].to(device)

        count = english.shape[0]

        output = transformer_model(english, german[:,:-1])
        
        output = output.reshape(-1, output.shape[2])
        german = german[:, 1:]
        german = german.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, german)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1)

        optimizer.step()
        
        if scheduler:
            scheduler.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
        
        loss_meter.update(loss.item(), count)
        bar.set_postfix(loss=loss_meter.avg, lr=get_lr(optimizer), step=step)
    
    checkpoint = {"state_dict": transformer_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(checkpoint, f"/pscratch/sd/j/josh-ee/tf/model_ckpt/no_scheduler_epoch{epoch}_checkpoint.pth.tar")

    # Example Generation (Greedy Decode)
    ex = test[random.randint(0, len(test))]
    sentence = ex['translation']['en']
    src_indexes = torch.tensor(text_transform_eng(sentence)).unsqueeze(0).to(device)    
    translated_sentence_idx = translate_seq(transformer_model, src_indexes, device=device, max_len=50)
    translated_sentence = [ger_vocab.get_itos()[i] for i in translated_sentence_idx]
    print(f"\nExample sentence: \n {sentence}\n")
    print(f"Original Translation : \n{ex['translation']['de']}\n")
    print(f"Generated Translation : \n {' '.join(translated_sentence[1:-1])}\n")
    
    del src_indexes, ex, sentence, translated_sentence_idx, translated_sentence, checkpoint
    torch.cuda.empty_cache()
    _ = gc.collect()


# ### Sample Beam Search Generation from Test Data
load_model = False
if load_model:
    transformer_model.load_state_dict(torch.load("og_epoch1_checkpoint.pth.tar", map_location=device)['state_dict'])


# Function to compute loss on the test set
def evaluate_model(model, dataloader, device, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_items = 0
    bar = tqdm(dataloader, total=math.ceil(len(train)/batch_size))
    with torch.no_grad():  # No need to track gradients for evaluation
        for idx, data in enumerate(bar):
            src = data["src"]
            tgt = data["tgt"]
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)  # Shift for teacher forcing
            
            # Forward pass
            output = model(src, tgt_input)
            output_dim = output.shape[-1]
            
            # Reshape output for calculating loss
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
            total_items += tgt_output.shape[0]

    return total_loss / total_items

# Define the loss criterion, typically CrossEntropyLoss for classification tasks
criterion = torch.nn.CrossEntropyLoss(ignore_index=transformer_model.tgt_pad_idx)

# Example usage
test_loss = evaluate_model(transformer_model, test_dataloader, device, criterion)
print(f"Average loss on the test set: {test_loss:.3f}")


for n in range(5):
    print(f"Example {n+1}\n")
    ex = test[random.randint(0, len(test))]
    sentence = ex['translation']['en']
    src_indexes = torch.tensor(text_transform_eng(sentence)).unsqueeze(0).to(device)    
    k = 3
    translated_sentence_ids = translate_seq_beam_search(transformer_model, src_indexes, k=k, device=device, max_len=50)
    translated_sentence_ids = sorted(translated_sentence_ids, key= lambda x: x[1], reverse=True)
    translations = [[ger_vocab.get_itos()[i] for i in translated_sentence[0]] for translated_sentence in translated_sentence_ids]
    print(f"English : {ex['translation']['en']}\n")
    print(f"German : {ex['translation']['de']}")
    print(f"German Translations generated:\n")
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


# ## Calculating Bleu Score
from torchtext.data.metrics import bleu_score

def calculate_bleu(data, model, device, max_len=50):
    tgts = []
    preds = []
    for datum in tqdm(data):
        src = datum['translation']["en"]
        tgt = datum['translation']["de"]
        src_idx = torch.tensor(text_transform_eng(src)).unsqueeze(0).to(device)
        pred_tgt = translate_seq(model, src_idx, device, max_len)
        pred_tgt = pred_tgt[1:-1]
        pred_sent = [ger_vocab.get_itos()[i] for i in pred_tgt]
        preds.append(pred_sent)
        tgts.append([tokenizer_ger(tgt.lower())])

    return bleu_score(preds, tgts) 

bleu = calculate_bleu(test, transformer_model, device)
print("BLEU Score Achieved :", bleu)



