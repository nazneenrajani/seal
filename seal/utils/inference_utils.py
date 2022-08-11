import pickle
from dataclasses import dataclass
import torch

@dataclass
class InferenceResults:
    """
    Class for storing embeddings and losses from running inference on a model.
    
    Fields:
    - embeddings: (num_examples x num_dimensions) tensor of last-layer embeddings
    - losses: (num_examples x 1) tensor of losses
    - outputs: optional (num_examples x num_classes) tensor of output logits
    - labels: optional (num_examples x 1) tensor of labels
    """
    
    embeddings: torch.Tensor
    losses: torch.Tensor
    outputs: torch.Tensor = None
    labels: torch.Tensor = None 

def saveResults(fname, results):
    with open(fname, 'wb+') as f:
        pickle.dump(results, f)