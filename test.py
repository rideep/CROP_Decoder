from transformers import GPT2ForTokenClassification, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from dataset import get_test_tokens, get_test_labels
from torch.nn.utils.rnn import pad_sequence

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device.type)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# num_labels is 9 because we have 9 NER tags
model = GPT2ForTokenClassification.from_pretrained('gpt2-trained-ner-model')
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
sliding_test_tokens = [sliding_window(tokenizer.encode(tokens), 1024) for tokens in get_test_tokens()]
sliding_test_tokens = [item for sublist in sliding_test_tokens for item in sublist]

# Print the length of the longest sequence in train_tokens
# print(max(len(tokens) for tokens in sliding_test_tokens))

# Convert the tokens to tensors
test_tokens = [torch.tensor(tokens) for tokens in sliding_test_tokens]

# Pad the sequences
test_tokens_padded = pad_sequence(test_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)

# Create attention masks
attention_masks = test_tokens_padded != tokenizer.pad_token_id

# Your padded sequences are your input_ids
input_ids = test_tokens_padded

# Convert your labels to tensors
sliding_train_labels = [sliding_window(labels, 1024) for labels in get_test_labels()]
sliding_train_labels = [item for sublist in sliding_train_labels for item in sublist]
labels = [torch.tensor(label) for label in sliding_train_labels]

# Pad the labels
labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

# Create a TensorDataset from the inputs and labels
dataset = TensorDataset(input_ids, attention_masks, labels_padded)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=8)

# Evaluation
model.eval()  # Set the model to evaluation mode
total_loss = 0
total_correct = 0
total_labels = 0

with torch.no_grad():  # Disable gradient calculations
    for batch in data_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        # Compute the number of correct predictions
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_correct += torch.sum(predictions == labels).item()
        total_labels += labels.numel()

avg_test_loss = total_loss / len(data_loader)
accuracy = total_correct / total_labels
print(f'Test loss: {avg_test_loss}, Accuracy: {accuracy}')