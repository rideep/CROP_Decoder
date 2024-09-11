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

from transformers import pipeline
import random

pipe = pipeline("text-generation", model="./tune_only_personv2")

instruction = """You are an excellent linguist. The task is to label person entities in the given sentence. Noted that if the given sentence does not contain any person entities, just output the same sentence, or surround the extracted entities by @@ and ## if there exist person entities."""

person_train_tokens = []
person_train_labels = []
for token, label in zip(get_train_tokens(), get_train_labels()):
    if 1 in label or 2 in label:
        person_train_tokens.append(token)
        person_train_labels.append(label)

person_test_tokens = []
person_test_labels = []
for token, label in zip(get_test_tokens(), get_test_labels()):
    if 1 in label or 2 in label:
        person_test_tokens.append(token)
        person_test_labels.append(label)

import re

def extract_person(text):
    pattern = r"@@(.*?)##"
    matches = re.findall(pattern, text)
    return matches

def get_persons_from_test_data(tokens, labels):
    persons = []
    for token, label in zip(tokens, labels):
        if label in (1, 2):
            persons.append(token)
    return persons

# print(person_test_tokens[:5])
# print(person_test_labels[:5])
print(f"we will be testing here for {len(person_test_tokens)}")

def create_output(tokens, labels):
    output = []
    for token, label in zip(tokens, labels):
        if label in (1, 2):
            output.append(f"@@{token}##")
        else:
            output.append(token)
    return ' '.join(output)

def create_training_data(train_tokens, train_labels):
    return [{"Input": ' '.join(tokens), "Output": create_output(tokens, labels)} for tokens, labels in zip(train_tokens, train_labels)]

training_data = create_training_data(person_train_tokens, person_train_labels)

def create_prompt(test_tokens):
    random_training_data = random.sample(training_data, k=3)
    print(random_training_data)
    prompt = instruction + '\n'
    for data in random_training_data:
        prompt += f"Input: {data['Input']}\nOutput: {data['Output']}\n###\n"
    prompt += "Input: " + ' '.join(test_tokens) + "\nOutput: "
    return prompt

correct_person_count = 0  # TP
predicted_person_count = 0  # TP + FP
person_count = 0  # TP + FN

for tokens, labels in zip(person_test_tokens, person_test_labels):
    prompt = create_prompt(tokens)
    # print(prompt)
    # exit()
    generated_text = pipe(prompt, max_new_tokens=len(tokens) + 20, do_sample=True, temperature=0.5)
    interested_text = generated_text[0]['generated_text']
    predicted_persons = extract_person(interested_text.replace(prompt, "").split("###")[0])
    actual_persons = get_persons_from_test_data(tokens, labels)
    print(f"Predicted Persons: {predicted_persons}")
    print(f"Actual Persons: {actual_persons}")
    
    person_count += len(actual_persons)
    predicted_person_count += len(predicted_persons)
    correct_person_count += len(set(predicted_persons) & set(actual_persons))

accuracy = correct_person_count / person_count
precision = correct_person_count / predicted_person_count
recall = correct_person_count / person_count
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy in predicting person is {accuracy}")
print(f"Precision in predicting person is {precision}")
print(f"Recall in predicting person is {recall}")
print(f"F1 Score in predicting person is {f1_score}")

    