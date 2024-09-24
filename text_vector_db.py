import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from datasets import load_dataset


# Define Transformer Model with BERT backbone
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6):
        super(SimpleTransformer, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(d_model, self.encoder.config.vocab_size)

    def forward(self, src, tgt):
        src_enc = self.encoder(src).last_hidden_state
        tgt_emb = self.encoder.embeddings(tgt)
        output = self.transformer(src_enc, tgt_emb)
        return self.fc_out(output)


# Tokenization function
def tokenize_function(examples, tokenizer):
    src = tokenizer(examples['translation']['en'], padding="max_length", truncation=True, return_tensors="pt")[
        'input_ids']
    tgt = tokenizer(examples['translation']['it'], padding="max_length", truncation=True, return_tensors="pt")[
        'input_ids']
    return {'src': src[0], 'tgt': tgt[0]}  # Extract tensors from the lists


# Custom collate function to pad and batch data properly
def collate_fn(batch):
    src = torch.stack([item['src'] for item in batch])
    tgt = torch.stack([item['tgt'] for item in batch])
    return {'src': src, 'tgt': tgt}


# Load Dataset and Tokenizer
def load_data():
    dataset = load_dataset('opus_books', 'en-it')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split the dataset into train and validation sets
    dataset = dataset['train'].train_test_split(test_size=0.1)

    # Apply tokenization
    train_ds = dataset['train'].map(lambda x: tokenize_function(x, tokenizer),
                                    remove_columns=dataset['train'].column_names)
    val_ds = dataset['test'].map(lambda x: tokenize_function(x, tokenizer), remove_columns=dataset['test'].column_names)

    # Create DataLoaders with custom collate function
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, tokenizer


# Train the Transformer
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer().to(device)
    train_loader, val_loader, tokenizer = load_data()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(5):  # Number of epochs
        model.train()
        for batch in train_loader:
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = loss_fn(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                src, tgt = batch['src'].to(device), batch['tgt'].to(device)
                output = model(src, tgt[:, :-1])
                loss = loss_fn(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


if __name__ == "__main__":
    train_model()
