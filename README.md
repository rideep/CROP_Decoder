# crop-decoder
To train the translation model, download the dataset from https://drive.google.com/file/d/1kEUAbJGOz34TVJktwTrHKBv0WcqRdlMm/view?usp=drive_link


## GPT Neo Workflow
1. Train the model first with person-only-train.py
2. Use the above trained model on one-shot-inference-neo.py (You can update the value of k random to make it one-shot vs zero-shot)

## TowerInstruct Workflow
1. Use one-shot-inference-tower.py
