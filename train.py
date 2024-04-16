from torchtext.data.utils import get_tokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import itertools
from torch.nn import Transformer

# This is very similar to the notebook but its optimized for training on A100 (40GB)
# That is also why its a .py file so it can run via SSH

iwslt_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = 'mps'  # For Apple Silicon GPUs

print('DEVICE:', device)

# Adjust up or down according to VRAM
BATCH_SIZE = 32 # 32 is good for 40GB 

# Initialize tokenizers
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')

def tokenize(batch):
    # Access nested 'translation' and then language specific texts
    en_texts = [item['en'] for item in batch['translation']]
    de_texts = [item['de'] for item in batch['translation']]
    # Apply tokenization and ensure each text is a list of tokens
    batch['tokenized_en'] = [list(map(str, tokenizer_en(text))) for text in en_texts]
    batch['tokenized_de'] = [list(map(str, tokenizer_de(text))) for text in de_texts]
    return batch

# Apply the tokenization function
iwslt_dataset = iwslt_dataset.map(tokenize, batched=True, batch_size=1000, num_proc=4)

# Extract tokenized data
en_tokenized_texts = iwslt_dataset['train']['tokenized_en']
de_tokenized_texts = iwslt_dataset['train']['tokenized_de']


def build_vocab(tokenized_texts, min_freq=1):
    # Flatten the list of tokens
    all_tokens = list(itertools.chain.from_iterable(tokenized_texts))
    
    # Count all tokens
    token_freqs = Counter(all_tokens)
    
    # Remove tokens below a certain frequency threshold
    vocab = {token: idx + 1 for idx, (token, freq) in enumerate(token_freqs.items()) if freq >= min_freq}
    vocab['<pad>'] = 0  # Pad token at index 0
    vocab['<unk>'] = len(vocab)  # Unknown token at the last index
    return vocab

# Build vocabularies
EN_VOCAB = build_vocab(en_tokenized_texts)
DE_VOCAB = build_vocab(de_tokenized_texts)


def collate_fn(batch):
    en_batch = [item['tokenized_en'] for item in batch]
    de_batch = [['<sos>'] + item['tokenized_de'] + ['<eos>'] for item in batch]
    
    en_indices = [[EN_VOCAB.get(token, EN_VOCAB['<unk>']) for token in sentence] for sentence in en_batch]
    de_indices = [[DE_VOCAB.get(token, DE_VOCAB['<unk>']) for token in sentence] for sentence in de_batch]
    
    en_tensor = pad_sequence([torch.tensor(seq) for seq in en_indices], padding_value=EN_VOCAB['<pad>'], batch_first=True)
    de_tensor = pad_sequence([torch.tensor(seq) for seq in de_indices], padding_value=DE_VOCAB['<pad>'], batch_first=True)
    
    return {'en': en_tensor.to(device), 'de': de_tensor.to(device)}


train_loader = DataLoader(iwslt_dataset['train'], batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(iwslt_dataset['validation'], batch_size=32, collate_fn=collate_fn)

# Initialize embedding layers
d_model = 512

en_embedding = torch.nn.Embedding(len(EN_VOCAB), d_model).to(device)
de_embedding = torch.nn.Embedding(len(DE_VOCAB), d_model).to(device)


model = Transformer(
    d_model=d_model,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048
).to(device)

output_projection = torch.nn.Linear(d_model, len(DE_VOCAB)).to(device)


optimizer = torch.optim.Adam(list(model.parameters()) + list(output_projection.parameters()), lr=1e-4)

# optimizer = torch.optim.Adam(
#     list(model.parameters()) + list(output_projection.parameters()),
#     lr=1e-4,  # This initial learning rate is a placeholder, it will be updated by the scheduler
#     betas=(0.9, 0.98),  # β1 and β2
#     eps=1e-9  # ϵ
# )

num_epochs = 5

for epoch in range(num_epochs):
    for batch in train_loader:
        src_tensor = batch['en']
        tgt_tensor = batch['de']
        
        optimizer.zero_grad()

        src = en_embedding(src_tensor)
        tgt = de_embedding(tgt_tensor)

        # Adjust the sequence length compatibility as before
        if src.shape[1] > tgt.shape[1]:
            src = src[:, :tgt.shape[1], :]
        elif src.shape[1] < tgt.shape[1]:
            pad_size = tgt.shape[1] - src.shape[1]
            src = torch.nn.functional.pad(src, (0, 0, 0, pad_size), value=EN_VOCAB['<pad>'])

        out = model(src, tgt)
        out = output_projection(out)  # Project output to the DE vocabulary size

        # New code for calculating loss, ignoring padding
        target_mask = (tgt_tensor != DE_VOCAB['<pad>']).view(-1)
        loss = torch.nn.functional.cross_entropy(out.view(-1, len(DE_VOCAB)), tgt_tensor.view(-1), reduction='none')
        loss = (loss * target_mask).sum() / target_mask.sum()

        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')


    # Save a checkpoint at the end of each epoch
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'output_projection_state_dict': output_projection.state_dict(),  # Correct key for output projection state
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_en': EN_VOCAB,
        'vocab_de': DE_VOCAB,
        'settings': {
            'd_model': d_model,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'output_vocab_size': len(DE_VOCAB)
        }
    }

    torch.save(checkpoint, f'model_ckpt/transformer_checkpoint_epoch{epoch}.pth')



