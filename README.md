# Cross-Lingual NER with decoder-based multilingual labeled sequence

# Abstract
Cross-lingual Named Entity Recognition (NER)
is crucial for multilingual natural language pro-
cessing tasks, allowing systems to identify named
entities for low resources langauges. This study
investigated the potential enhancement of cross-
lingual NER performance through the implemen-
tation of a decoder-only transformer in conjunc-
tion with multilingual labeled sequence translation.
The decoder-only transformer architecture, known
for its simplicity and efficiency, is hypothesized
to improve the model’s ability to capture linguis-
tic features relevant to NER across languages. By
leveraging multilingual labeled sequence transla-
tion, the model learns to align representations of
named entities across languages, facilitating the
effective cross-lingual transfer
# Dataset
CCaligned, CoNLL-5, and XTREME-40 datasets
# Environment
Python: >= 3.6
PyTorch: >= 1.5.0
# Training
transalation
To train the translation model, download the dataset from https://drive.google.com/file/d/1kEUAbJGOz34TVJktwTrHKBv0WcqRdlMm/view?usp=drive_link


## GPT Neo Workflow
1. Train the model first with person-only-train.py
2. Use the above trained model on one-shot-inference-neo.py (You can update the value of k random to make it one-shot vs zero-shot)

## TowerInstruct Workflow
1. Use one-shot-inference-tower.py
