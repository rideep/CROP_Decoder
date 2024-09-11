# create training dataset
from datasets import load_dataset

english_dataset = load_dataset("conll2003")

train_dataset = english_dataset['train'].remove_columns(['pos_tags'])
test_dataset = english_dataset['test'].remove_columns(['pos_tags'])
validation_dataset = english_dataset['validation'].remove_columns(['pos_tags'])

def get_test_tokens():
    tokens = [tokens for tokens in test_dataset["tokens"] if tokens]
    return tokens

def get_test_labels():
    labels = [labels for labels in test_dataset["ner_tags"] if labels]
    return labels

def get_validation_tokens():
    return [tokens for tokens in validation_dataset["tokens"] if tokens]

def get_validation_labels():
    return [labels for labels in validation_dataset["ner_tags"] if labels]

def get_train_tokens():
    return [tokens for tokens in train_dataset["tokens"] if tokens]

def get_train_labels():
    return [labels for labels in train_dataset["ner_tags"] if labels]

def create_output(tokens, labels):
    output = []
    for token, label in zip(tokens, labels):
        if label in (1, 2):
            output.append(f"@@{token}##")
        else:
            output.append(token)
    return ' '.join(output)

instruction = """You are an excellent linguist. The task is to label person entities in the given sentence. Noted that if the given sentence does not contain any person entities, just output the same sentence, or surround the extracted entities by @@ and ## if there exist person entities."""

with open('train.txt', 'w') as f:
    f.write(instruction + "\n")
    for token, label in zip(get_train_tokens(), get_train_labels()):
        if (1 in label and 2 in label):
            f.write(f"Input: {' '.join(token)}\n" + f"Output: {create_output(token, label)}\n" + "###\n")

# exit()
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM
from transformers import Trainer, TrainingArguments

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

import torch 
def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)        
    model = GPTNeoForCausalLM.from_pretrained(model_name)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
    )
        
    trainer.train()
    trainer.save_model()

# you need to set parameters 
train_file_path = "train.txt"
model_name = "EleutherAI/gpt-neo-125m"
output_dir = './final_neo_people_only_model'
overwrite_output_dir = False
per_device_train_batch_size = 16
num_train_epochs = 50.0
save_steps = 500

print("Training is started")
# it takes about 30 minutes to train in colab.
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)

print("Training is done")