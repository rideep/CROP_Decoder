from transformers import GPT2Tokenizer, GPT2ForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from dataset import get_train_tokens, get_train_labels
import torch
from torch.nn.utils.rnn import pad_sequence

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device.type)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# num_labels is 9 because we have 9 NER tags
model = GPT2ForTokenClassification.from_pretrained('gpt2', num_labels=9)
print("Model and tokenizer loaded")

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move the model to the GPU if available
model = model.to(device)

# Because there is a sequence that is longer than 1024 tokens, you need to split it into smaller sequences that fit within the model's maximum sequence length. One way to do this is to use a sliding window approach to create overlapping sequences of a fixed length (e.g., 1024 tokens) from the original sequence.
def sliding_window(tokens, n):
    if len(tokens) > n:
        return [tokens[i:i+n] for i in range(len(tokens) - n + 1)]
    else:
        return [tokens]

# Tokenize the training data
sliding_train_tokens = [sliding_window(tokenizer.encode(tokens), 1024) for tokens in get_train_tokens()]
sliding_train_tokens = [item for sublist in sliding_train_tokens for item in sublist]

# Print the length of the longest sequence in train_tokens
print(max(len(tokens) for tokens in sliding_train_tokens))

# Convert the tokens to tensors
train_tokens = [torch.tensor(tokens) for tokens in sliding_train_tokens]

# Pad the sequences
train_tokens_padded = pad_sequence(train_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)

# Create attention masks
attention_masks = train_tokens_padded != tokenizer.pad_token_id

# Your padded sequences are your input_ids
input_ids = train_tokens_padded

# Convert your labels to tensors
sliding_train_labels = [sliding_window(labels, 1024) for labels in get_train_labels()]
sliding_train_labels = [item for sublist in sliding_train_labels for item in sublist]
labels = [torch.tensor(label) for label in sliding_train_labels]

# Pad the labels
labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

# Create a TensorDataset from the inputs and labels
dataset = TensorDataset(input_ids, attention_masks, labels_padded)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=8)

# Fine-tune the model
model.train()
optim = AdamW(model.parameters(), lr=5e-5)

# Define the number of epochs
epochs = 3

# Training loop
for epoch in range(epochs):
    total_loss = 0  
    for batch in data_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(labels.shape)

        # print(torch.max(labels))

        # print(f"Max position embeddings: {model.config.max_position_embeddings}")
        # print(f"Max input sequence length: {input_ids.shape[1]}")

        optim.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # print(outputs.logits.shape)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optim.step()

    print(f'Epoch {epoch + 1}/{epochs} done')
    # Compute the average loss over the epoch
    avg_train_loss = total_loss / len(data_loader)
    print(f'Loss after epoch {epoch + 1}: {avg_train_loss}')

# Save the model
model.save_pretrained('gpt2-trained-ner-model')