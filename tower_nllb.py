test_cases = [
    {
        "source_text": "राम आज स्कुल गएका छन्।",
        "target_text": "Ram went to school today.",
        "ner": ["राम"]
    },
    {
        "source_text": "यो सीता को किताब हो।",
        "target_text": "This is Sita's book.",
        "ner": ["सीता"]
    },
    {
        "source_text": "हरी र गीता ले पार्टी मा भेटे।",
        "target_text": "Hari and Gita met at the party.",
        "ner": ["हरी", "गीता"]
    },
    {
        "source_text": "उहाँको नाम राजेश हो।",
        "target_text": "His name is Rajesh.",
        "ner": ["राजेश"]
    },
    {
        "source_text": "सबिता ले गीत गाइन्।",
        "target_text": "Sabita sang a song.",
        "ner": ["सबिता"]
    },
    {
        "source_text": "कृष्ण सँग कुरा गर्नुभयो?",
        "target_text": "Did you talk with Krishna?",
        "ner": ["कृष्ण"]
    },
    {
        "source_text": "बिना भोलि आउने भएकी छिन्।",
        "target_text": "Bina is coming tomorrow.",
        "ner": ["बिना"]
    },
    {
        "source_text": "अनिल त्यहाँ बस्छन्।",
        "target_text": "Anil lives there.",
        "ner": ["अनिल"]
    },
    {
        "source_text": "मीना र करन कलेज जाँदैछन्।",
        "target_text": "Meena and Karan are going to college.",
        "ner": ["मीना", "करन"]
    },
    {
        "source_text": "प्रकाश ले मलाई फोन गर्ने भयो।",
        "target_text": "Prakash said he would call me.",
        "ner": ["प्रकाश"]
    },
    {
        "source_text": "सुनीता को जन्मदिन अर्को हप्ता हो।",
        "target_text": "Sunita's birthday is next week.",
        "ner": ["सुनीता"]
    },
    {
        "source_text": "दिपेश को कार नयाँ छ।",
        "target_text": "Dipesh's car is new.",
        "ner": ["दिपेश"]
    },
    {
        "source_text": "ज्ञानेन्द्र ले पुस्तक पढे।",
        "target_text": "Gyanendra read a book.",
        "ner": ["ज्ञानेन्द्र"]
    },
    {
        "source_text": "आशा ले खाना पकाएकी छिन्।",
        "target_text": "Asha has cooked food.",
        "ner": ["आशा"]
    },
    {
        "source_text": "बिपिन र अर्जुन ले फुटबल खेले।",
        "target_text": "Bipin and Arjun played football.",
        "ner": ["बिपिन", "अर्जुन"]
    },
    {
        "source_text": "समीर ले किताब किने।",
        "target_text": "Sameer bought a book.",
        "ner": ["समीर"]
    },
    {
        "source_text": "निर्मल बजार जाँदै छ।",
        "target_text": "Nirmal is going to the market.",
        "ner": ["निर्मल"]
    },
    {
        "source_text": "पूजा ले नृत्य गरिन्।",
        "target_text": "Pooja danced.",
        "ner": ["पूजा"]
    },
    {
        "source_text": "संजीव ले मलाई सहयोग गर्‍यो।",
        "target_text": "Sanjeev helped me.",
        "ner": ["संजीव"]
    },
    {
        "source_text": "बिनोद र रिता फिल्म हेर्न गए।",
        "target_text": "Binod and Rita went to watch a movie.",
        "ner": ["बिनोद", "रिता"]
    }
]

from transformers import pipeline

def translation(text, source='npi_Deva', target='eng_Latn'):
    model_name = 'facebook/nllb-200-distilled-600M'

    translator = pipeline('translation', model=model_name, tokenizer=model_name, src_lang=source, tgt_lang=target)
    output = translator(text, max_length=400)

    output = output[0]['translation_text']
    return output

correct_person_count = 0  # TP
predicted_person_count = 0  # TP + FP
person_count = 0  # TP + FN

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")

import ast

def get_answer(messages):
    count = 0
    while count < 5:
        count += 1
        try:
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
            answer = outputs[0]["generated_text"]
            # print(answer)
            answer = answer.split("<|im_start|>assistant")[1]
            return ast.literal_eval(answer)
        except Exception as e:
            print(f"Error: {e}")

for item in test_cases:
    source_text = item['source_text']
    target_text = item['target_text']
    expected_ner = item['ner']

    translated_text = translation(source_text)
    print(f"Translated text: {translated_text}")
    
    messages = [
        { "role": "user", "content": f"Study this taxonomy for classifying named entities:\n- Person - Names of people\n- Group - Groups of people, organizations, corporations or other entities\n- CreativeWorks - Titles of creative works like movie, song, and book titles\n- Location - Location or physical facilities\n- Medical - Entities from the medical domain, including diseases, symptoms, and medications\n- Product - Consumer products such as food, drinks, clothing, and vehicles. Identify all named entities in the following tokens:\n{translated_text.split(' ')}\nAdditionally, you should add B- to the first token of a given entity and I- to subsequent ones if they exist. For tokens that are not named entities, mark them as O.\nAnswer: " },
    ]
    
    # print(messages)
    
    eval_answer = get_answer(messages)
    print(f"Evaluated Answer: {eval_answer}")
    if eval_answer is None:
        continue
    # [('EU', 'B-Group'), ('rejects', 'O'), ('German', 'B-Location'), ('call', 'O'), ('to', 'O'), ('boycott', 'O'), ('British', 'B-Product'), ('lamb', 'I-Product'), ('.', 'O')]
    predicted_ner = [translation(entity[0], source = 'eng_Latn', target = 'npi_Deva') for entity in eval_answer if entity[1] in ['B-Person', 'I-Person']]
    print(f"Expected NER: {expected_ner}")
    print(f"Predicted NER: {predicted_ner}")
    for p_ner in predicted_ner:
        if p_ner in expected_ner:
            ner_start = source_text.find(p_ner)
            ner_end = ner_start + len(p_ner)
            correct_person_count += 1
            print(f"NER: {p_ner}, Start: {ner_start}, End: {ner_end}")
        predicted_person_count += 1
    person_count += len(expected_ner)
    
accuracy = correct_person_count / person_count
precision = correct_person_count / predicted_person_count
recall = correct_person_count / person_count
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy in predicting person is {accuracy}")
print(f"Precision in predicting person is {precision}")
print(f"Recall in predicting person is {recall}")
print(f"F1 Score in predicting person is {f1_score}")