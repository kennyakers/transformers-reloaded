import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn import Transformer
from datasets import load_dataset
import itertools

# This is very similar to the notebook but its optimized for training on A100 (40GB)
# That is also why its a .py file so it can run via SSH

iwslt_dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = 'mps'  # For Apple Silicon GPUs


# Adjust up or down according to VRAM
BATCH_SIZE = 32 # 32 is good for 40GB 

# Initialize tokenizers
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')

def tokenize(batch):
    en_texts = [item['en'] for item in batch['translation']]
    de_texts = [item['de'] for item in batch['translation']]
    batch['tokenized_en'] = [list(map(str, tokenizer_en(text))) for text in en_texts]
    batch['tokenized_de'] = [list(map(str, tokenizer_de(text))) for text in de_texts]
    return batch

# Tokenize the data
iwslt_dataset = iwslt_dataset.map(tokenize, batched=True, batch_size=1000, num_proc=4)

# Function to load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    d_model=checkpoint['settings']['d_model']
    model = Transformer(
        d_model=d_model,
        nhead=checkpoint['settings']['nhead'],
        num_encoder_layers=checkpoint['settings']['num_encoder_layers'],
        num_decoder_layers=checkpoint['settings']['num_decoder_layers'],
        dim_feedforward=checkpoint['settings']['dim_feedforward']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    output_projection = torch.nn.Linear(
        checkpoint['settings']['d_model'], checkpoint['settings']['output_vocab_size']
    )
    # Ensure to load the state dict for the linear module correctly
    if 'output_projection_state_dict' in checkpoint:
        output_projection.load_state_dict(checkpoint['output_projection_state_dict'])
    else:
        print("No saved state_dict for output_projection found in checkpoint.")

    vocab_en = checkpoint['vocab_en']
    vocab_de = checkpoint['vocab_de']
    
    return model, output_projection, vocab_en, vocab_de, d_model


model, output_projection, EN_VOCAB, DE_VOCAB, d_model = load_checkpoint('model_ckpt/transformer_checkpoint_epoch4.pth')
output_projection = output_projection.to(device)

en_embedding = torch.nn.Embedding(len(EN_VOCAB), d_model).to(device)
de_embedding = torch.nn.Embedding(len(DE_VOCAB), d_model).to(device)

# Function for collating batches
def collate_fn(batch):
    en_batch = [item['tokenized_en'] for item in batch]
    de_batch = [['<sos>'] + item['tokenized_de'] + ['<eos>'] for item in batch]
    en_indices = [[EN_VOCAB.get(token, EN_VOCAB['<unk>']) for token in sentence] for sentence in en_batch]
    de_indices = [[DE_VOCAB.get(token, DE_VOCAB['<unk>']) for token in sentence] for sentence in de_batch]
    en_tensor = pad_sequence([torch.tensor(seq) for seq in en_indices], padding_value=EN_VOCAB['<pad>'], batch_first=True)
    de_tensor = pad_sequence([torch.tensor(seq) for seq in de_indices], padding_value=DE_VOCAB['<pad>'], batch_first=True)
    return {'en': en_tensor.to(device), 'de': de_tensor.to(device)}

# Validation DataLoader
test_loader = DataLoader(iwslt_dataset['test'], batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Evaluate the model
model.eval()
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        src_tensor = batch['en'] 
        tgt_tensor = batch['de'] 

        src = en_embedding(src_tensor)
        tgt = de_embedding(tgt_tensor)

        # Adjust the sequence length compatibility as before
        if src.shape[1] > tgt.shape[1]:
            src = src[:, :tgt.shape[1], :]
        elif src.shape[1] < tgt.shape[1]:
            pad_size = tgt.shape[1] - src.shape[1]
            src = torch.nn.functional.pad(src, (0, 0, 0, pad_size), value=EN_VOCAB['<pad>'])

        out = model(src, tgt)
        out = output_projection(out) 
        
        target_mask = (tgt_tensor != DE_VOCAB['<pad>']).view(-1)
        loss = torch.nn.functional.cross_entropy(out.view(-1, len(DE_VOCAB)), tgt_tensor.view(-1), reduction='none')
        loss = (loss * target_mask).sum() / target_mask.sum()
        total_loss += loss.item()

    print(f'Average Validation Loss: {total_loss / len(test_loader)}')
