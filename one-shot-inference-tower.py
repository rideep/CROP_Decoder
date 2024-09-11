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

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")

correct_person_count = 0
person_count = 0
false_positives = 0
false_negatives = 0

person_test_tokens = []
person_test_labels = []
for token, label in zip(get_test_tokens(), get_test_labels()):
    if 1 in label or 2 in label:
        person_test_tokens.append(token)
        person_test_labels.append(label)

retry_tokens = []
retry_labels = []

def test_model(local_test_tokens, local_test_labels, count = 0):
    global correct_person_count
    global false_positives
    global false_negatives
    global person_count
    retry_tokens.append([])
    retry_labels.append([])
    for tokens, labels in zip(local_test_tokens, local_test_labels):
        messages = [
            { "role": "user", "content": f"Study this taxonomy for classifying named entities:\n- Person - Names of people\n- Group - Groups of people, organizations, corporations or other entities\n- CreativeWorks - Titles of creative works like movie, song, and book titles\n- Location - Location or physical facilities\n- Medical - Entities from the medical domain, including diseases, symptoms, and medications\n- Product - Consumer products such as food, drinks, clothing, and vehicles. Identify all named entities in the following tokens:\n{tokens}\nAdditionally, you should add B- to the first token of a given entity and I- to subsequent ones if they exist. For tokens that are not named entities, mark them as O.\nAnswer: " },
        ]
        # print(messages)
        # exit()
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
        answer = outputs[0]["generated_text"]
        print(answer)
        answer = answer.split("<|im_start|>assistant")[1]

        import ast
        try:
            eval_answer = ast.literal_eval(answer)
            # [('EU', 'B-Group'), ('rejects', 'O'), ('German', 'B-Location'), ('call', 'O'), ('to', 'O'), ('boycott', 'O'), ('British', 'B-Product'), ('lamb', 'I-Product'), ('.', 'O')]
            for item, true_label in zip(eval_answer, labels):
                token, predicted_label = item
                if true_label == 2 or true_label == 1:
                    if predicted_label == 'B-Person' or predicted_label == 'I-Person':
                        correct_person_count += 1
                    if predicted_label != 'B-Person' and predicted_label != 'I-Person':
                        false_negatives += 1
                    person_count += 1
                elif (predicted_label == 'B-Person' or predicted_label == 'I-Person') and (true_label != 2 and true_label != 1):
                    false_positives += 1
        except Exception as e:
            print(f"Error: {e}")
            retry_tokens[count].append(tokens)
            retry_labels[count].append(labels)
    
test_model(person_test_tokens, person_test_labels)

iter = 0
while retry_tokens and retry_labels and iter < 5:
    print(f"Retrying iteration {iter + 1}")
    print(f"Retrying for {len(retry_tokens[iter])} tokens")
    test_model(retry_tokens[iter], retry_labels[iter], count=iter+1)
    iter += 1

total_predictions = person_count

accuracy = correct_person_count / total_predictions
precision = correct_person_count / (correct_person_count + false_positives)
recall = correct_person_count / (correct_person_count + false_negatives)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
f1_score = 2 * (precision * recall) / (precision + recall)
print(f"F1 Score: {f1_score}")

print(f"iterations: {iter}")
print(len(retry_tokens[iter]))
