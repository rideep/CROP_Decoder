test_cases = [
    {
        "source_text": "রবার্ট কালেজে যাচ্ছে।",
        "target_text": "Robert is going to college.",
        "ner": ["Robert"]
    },
    {
        "source_text": "মারিয়া আমার ভাল বন্ধু।",
        "target_text": "Maria is my good friend.",
        "ner": ["Maria"]
    },
    {
        "source_text": "জন এবং জেসিকা পার্কে গেল।",
        "target_text": "John and Jessica went to the park.",
        "ner": ["John", "Jessica"]
    },
    {
        "source_text": "ডেভিড কি বই পড়ছে?",
        "target_text": "Is David reading a book?",
        "ner": ["David"]
    },
    {
        "source_text": "এলিস আজ খুব খুশি।",
        "target_text": "Alice is very happy today.",
        "ner": ["Alice"]
    },
    {
        "source_text": "মাইকেল কোথায়?",
        "target_text": "Where is Michael?",
        "ner": ["Michael"]
    },
    {
        "source_text": "সারাহ এবং পল বিয়ে করেছে।",
        "target_text": "Sarah and Paul got married.",
        "ner": ["Sarah", "Paul"]
    },
    {
        "source_text": "টম বাজারে যাচ্ছে।",
        "target_text": "Tom is going to the market.",
        "ner": ["Tom"]
    },
    {
        "source_text": "লিজা একজন ডাক্তার।",
        "target_text": "Lisa is a doctor.",
        "ner": ["Lisa"]
    },
    {
        "source_text": "স্টিভ কি গিটার বাজাতে পারে?",
        "target_text": "Can Steve play the guitar?",
        "ner": ["Steve"]
    },
    {
        "source_text": "জুলিয়া গান গাইছে।",
        "target_text": "Julia is singing a song.",
        "ner": ["Julia"]
    },
    {
        "source_text": "নিক এবং জুডি সিনেমা দেখতে যাচ্ছে।",
        "target_text": "Nick and Judy are going to watch a movie.",
        "ner": ["Nick", "Judy"]
    },
    {
        "source_text": "হেনরি নতুন গাড়ি কিনেছে।",
        "target_text": "Henry bought a new car.",
        "ner": ["Henry"]
    },
    {
        "source_text": "স্যামান্থা রান্না করতে পছন্দ করে।",
        "target_text": "Samantha likes to cook.",
        "ner": ["Samantha"]
    },
    {
        "source_text": "এডওয়ার্ড কি চিঠি লিখেছে?",
        "target_text": "Did Edward write the letter?",
        "ner": ["Edward"]
    },
    {
        "source_text": "গ্রেস কলকাতায় থাকে।",
        "target_text": "Grace lives in Kolkata.",
        "ner": ["Grace"]
    },
    {
        "source_text": "এথান বিজ্ঞানে ভালো।",
        "target_text": "Ethan is good at science.",
        "ner": ["Ethan"]
    },
    {
        "source_text": "ওলিভিয়া এবং লুক গল্প বলছে।",
        "target_text": "Olivia and Luke are telling stories.",
        "ner": ["Olivia", "Luke"]
    },
    {
        "source_text": "রাহেল লাইব্রেরিতে কাজ করে।",
        "target_text": "Rachel works at the library.",
        "ner": ["Rachel"]
    },
    {
        "source_text": "ম্যাথিউ সাঁতার কাটতে যাচ্ছে।",
        "target_text": "Matthew is going swimming.",
        "ner": ["Matthew"]
    }
]


from transformers import pipeline

def translation(text, source='ben_Beng', target='eng_Latn'):
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
    predicted_ner = [entity[0] for entity in eval_answer if entity[1] in ['B-Person', 'I-Person']]
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