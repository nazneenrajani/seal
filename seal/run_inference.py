from unittest import result
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

import os
import numpy as np
from tqdm import tqdm

from utils.inference_utils import InferenceResults, saveResults

# Load validation set

def load_session(dataset, model, split):
    dataset = load_dataset(dataset, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=256, drop_last=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer, dataloader, model

# Add hook to capture hidden layer
def get_input(name, model):
    hidden_layers = {}
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook, hidden_layers

def run_inference(dataset='yelp_polarity', model='textattack/albert-base-v2-yelp-polarity', split='test', output_path='./assets/data/inference_results'):
    tokenizer, dataloader, model = load_session(dataset,model,split)
    model.eval()
    model.to('cpu')
    hook, hidden_layers = model.classifier.register_forward_hook(get_input('last_layer', model))
    # Run inference on entire dataset
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
    #outputs = torch.hstack(output_list)
    losses = torch.hstack(loss_list)
    targets = torch.hstack(labels)
    #inputs = torch.hstack(example)
    results = save_results(embeddings,losses,targets)
    saveResults(os.path.join(output_path,dataset+'.pkl'),results)



def save_results(embeddings, losses, labels):
    results = InferenceResults(
        embeddings = torch.clone(embeddings),
        losses     = losses,
        labels     = labels
    )
    return results
    

