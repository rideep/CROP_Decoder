import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import nltk
from gensim.models import Word2Vec
from nltk.corpus import brown
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download NLTK data
nltk.download('brown')

# Load Brown corpus for training Word2Vec model
sentences = brown.sents()

# Train Word2Vec model
word_vectors = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Define the hyperparameters
INPUT_DIM = len(word_vectors.wv.key_to_index)
OUTPUT_DIM = len(word_vectors.wv.key_to_index)
EMB_DIM = 100
HID_DIM = 512
DROPOUT = 0.5

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors.wv.vectors))
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.long()  # Convert input to torch.long
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Define the Seq2Seq class which utilizes only the Decoder
class Seq2Seq(nn.Module):
    def __init__(self, decoder, device):
        super().__init__()
        self.decoder = decoder
        self.device = device

    def forward(self, trg, hidden, cell):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = OUTPUT_DIM

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        input = trg[0,:]

        # Initialize hidden and cell if they are None
        if hidden is None:
            hidden = torch.zeros(1, batch_size, HID_DIM).to(self.device)
        if cell is None:
            cell = torch.zeros(1, batch_size, HID_DIM).to(self.device)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1

        return outputs

# Initialize decoder
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DROPOUT)

# Create the Seq2Seq model with only the decoder
model = Seq2Seq(decoder, device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assume padding index is 0

# Function to evaluate the model and calculate loss
def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    translated_sentences = []
    with torch.no_grad():
        for trg, src in iterator:
            src = src.to(device)
            trg = trg.to(device)
            output = model(trg, None, None)
            output_dim = output.shape[-1]

            # Calculate the loss per timestep
            max_trg_len = trg.shape[0]
            loss = 0
            for t in range(1, max_trg_len):
                output_seq = output[t].unsqueeze(0)  # Add batch dimension
                trg_seq = trg[t].unsqueeze(0)        # Add batch dimension
                
                # Ensure target tensor has the same vocabulary size as the model's output
                trg_seq = trg_seq[:, :output_dim]
                
                loss += criterion(output_seq.squeeze(0), trg_seq.squeeze(0))
            
            epoch_loss += loss.item()
            output = output.argmax(-1)
            if output.dim() == 1:
                output_sentence = [list(word_vectors.wv.index_to_key[token.item()]) for token in output]
                output_sentence = ' '.join(output_sentence)
                translated_sentences.append(output_sentence)
            else:
                for token_seq in output:
                    output_sentence = [word_vectors.wv.index_to_key[token.item()] for token in token_seq]
                    output_sentence = ' '.join(map(str, output_sentence))  # Convert tokens to string before joining
                    translated_sentences.append(output_sentence)

    return epoch_loss / len(iterator), translated_sentences

# Function to train and test the model
def train_and_test_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, clip, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for trg, src in train_loader:
            # print("Target Sentences: ")
            # print(trg)
            # print("Source Sentences: ")
            # print(src)
            # Convert source and target sequences to tensors
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            output = model(trg, None, None)  # Swapped trg and src here
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            if output.size(0) != trg.size(0):
                min_size = min(output.size(0), trg.size(0))
                output = output[:min_size]
                trg = trg[:min_size]

            loss = criterion(output, trg)
            # Calculate the loss
            # loss = 0
            # for t in range(1, trg.shape[0]):
            #     output_seq = output[t].view(-1, output_dim)
            #     trg_seq = trg[t].view(-1)
            #     loss += criterion(output_seq, trg_seq)


            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

        valid_loss, _ = evaluate_model(model, valid_loader, criterion)

        print(f'Epoch: {epoch + 1:02} | Train Loss: {epoch_loss / len(train_loader):.3f} | Val. Loss: {valid_loss:.3f}')
    test_loss, translated_outputs = evaluate_model(model, test_loader, criterion)
    print(f'| Test Loss: {test_loss:.3f}')
    with open('translated_output.txt', 'w', encoding='utf-8') as file:
        for output in translated_outputs:
            file.write(output + '\n')
    torch.save(model.state_dict(), 'model_translate.pt')
    print("Model and Output file (\"translated_output.txt\") saved Successfully.")

# Define collate function for DataLoader
def collate_fn(data):
    src_seqs, trg_seqs = zip(*data)
    
    # Pad source sequences
    padded_src_seqs = nn.utils.rnn.pad_sequence(
        [torch.tensor([word_vectors.wv.key_to_index[word] for word in seq if word in word_vectors.wv.key_to_index]) for seq in src_seqs],
        batch_first=True,
        padding_value=0  # Pad with 0 (assuming padding index is 0)
    )
    
    # Pad target sequences
    padded_trg_seqs = nn.utils.rnn.pad_sequence(
        [torch.tensor([word_vectors.wv.key_to_index[word] for word in seq if word in word_vectors.wv.key_to_index]) for seq in trg_seqs],
        batch_first=True,
        padding_value=0  # Pad with 0 (assuming padding index is 0)
    )
    
    return padded_src_seqs, padded_trg_seqs



def main(model):
    # Load the dataset from text file and format it appropriately
    with open('data/ben.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            english, bengali, *_ = line.split('\t')
            data.append((english.strip(), bengali.strip()))

    # Define the percentages for train, validation, and test sets
    train_percentage = 0.85
    valid_percentage = 0.1
    test_percentage = 0.05

    # Calculate the number of examples for each set
    total_examples = len(data)
    train_size = int(train_percentage * total_examples)
    valid_size = int(valid_percentage * total_examples)
    test_size = total_examples - train_size - valid_size

    # Shuffle the data
    random.shuffle(data)

    # Split data into train, validation, and test sets
    train_data = data[:train_size]
    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:]

    # Set up the data loaders
    BATCH_SIZE = 4
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Move model to the appropriate device
    model = model.to(device)

    # Train and test the model
    N_EPOCHS = 10000
    CLIP = 1
    train_and_test_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, CLIP, N_EPOCHS)

    print(f"Model Trained and Evaluated successfully.")

if __name__ == '__main__':
    # Initialize the model
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DROPOUT)
    model = Seq2Seq(decoder, device)

    # Pass the model to the main function
    main(model)