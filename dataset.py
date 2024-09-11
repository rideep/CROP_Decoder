from datasets import load_dataset, concatenate_datasets

english_dataset = load_dataset("conll2003")
spanish_dataset = load_dataset("conll2002", "es")
dutish_dataset = load_dataset("conll2002", "nl")

english_train_dataset = english_dataset['train'].remove_columns(['pos_tags'])
spanish_train_dataset = spanish_dataset['train'].remove_columns(['pos_tags'])
dutish_train_dataset = dutish_dataset['train'].remove_columns(['pos_tags'])

english_test_dataset = english_dataset['test'].remove_columns(['pos_tags'])
spanish_test_dataset = spanish_dataset['test'].remove_columns(['pos_tags'])
dutish_test_dataset = dutish_dataset['test'].remove_columns(['pos_tags'])

english_validation_dataset = english_dataset['validation'].remove_columns(['pos_tags'])
spanish_validation_dataset = spanish_dataset['validation'].remove_columns(['pos_tags'])
dutish_validation_dataset = dutish_dataset['validation'].remove_columns(['pos_tags'])


# Concatenate the 'train' splits of the datasets
train_dataset = concatenate_datasets([english_train_dataset, spanish_train_dataset, dutish_train_dataset])

# Concatenate the 'validation' splits of the datasets
validation_dataset = concatenate_datasets([english_validation_dataset, spanish_validation_dataset, dutish_validation_dataset])

# Concatenate the 'test' splits of the datasets
test_dataset = concatenate_datasets([english_test_dataset, spanish_test_dataset, dutish_test_dataset])

def get_test_tokens():
    tokens = [tokens for tokens in test_dataset["tokens"] if tokens]
    return tokens

def get_test_labels():
    labels = [labels for labels in test_dataset["ner_tags"] if labels]
    return labels

def get_train_tokens():
    return [tokens for tokens in train_dataset["tokens"] if tokens]

def get_train_labels():
    return [labels for labels in train_dataset["ner_tags"] if labels]