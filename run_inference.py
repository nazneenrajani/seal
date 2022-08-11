import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

import os
import numpy as np
from tqdm import tqdm

from utils.inference_utils import InferenceResults, saveResults

# Note: we load cached copies of the dataset, tokenizer, and model to make inference work without an internet connection
""" data_dir = os.environ['DATA_DIR'] 
amazon_dir = os.path.join(data_dir, 'amazon')
model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'amazon') """

# Load validation set
print('Loading dataset...')
dataset = load_dataset("ag_news", split='test')
dataloader = DataLoader(
    dataset,
    #Subset(dataset['test'], range(20480)),
    batch_size=256, drop_last=True
)

# Load tokenizer + model
print('Loading model...')

#tokenizer = AutoTokenizer.from_pretrained("roberta-base-bne-finetuned-amazon_reviews_multi", truncation=True)
#tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-yelp-polarity")
#tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")
#model = AutoModelForSequenceClassification.from_pretrained("roberta-base-bne-finetuned-amazon_reviews_multi")
#model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-yelp-polarity")
#model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")
model.eval()
model.to('cpu')

# Add hook to capture hidden layer
hidden_layers = {}
def get_input(name):
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook
model.classifier.register_forward_hook(get_input('last_layer'))

# Run inference on entire dataset
print('Running inference...')
hidden_list = []
loss_list = []
output_list = []
example = []
labels = []
criterion = nn.CrossEntropyLoss(reduction='none')
softmax = nn.Softmax(dim=1)
with torch.no_grad():
    for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True):
        batch_ex = [ex[:512] for ex in batch['text']]
        inputs = tokenizer(batch_ex, padding=True, return_tensors='pt').to('cpu')
        targets = batch['label']
        
        outputs = model(**inputs)['logits']
        loss = criterion(outputs, targets)
        predictions = softmax(outputs)
        
        hidden_list.append(hidden_layers['last_layer'].cpu())
        loss_list.append(loss.cpu())
        #output_list.append(predictions[:, 1].cpu())
        output_list.append(np.argmax(predictions, axis=1))
        labels.append(targets)
        example.append(inputs['input_ids'])


embeddings = torch.vstack(hidden_list)
outputs = torch.hstack(output_list)
losses = torch.hstack(loss_list)
targets = torch.hstack(labels)
inputs = torch.hstack(example)

results = InferenceResults(
    aux = inputs,
    embeddings = torch.clone(embeddings),
    outputs    = outputs,
    losses     = losses,
    labels     = targets,
)
saveResults(
    os.path.join('../assets/data/inference_results', 'agnews_test_minibert.pkl'),
    results
)

