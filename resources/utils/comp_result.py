import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle

def evaluate(model, X, y):
    model.eval()
    
    if len(X) == 0:
        return np.nan
            
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32)
        preds = torch.argmax(model(X), dim=1).numpy()
    return accuracy_score(y, preds)

def pad_history(history, class_order, total_steps):
    """
    Allinea history inserendo np.nan negli step
    in cui una classe non è ancora presente.
    """
    padded = {}

    for cls in class_order:
        accs = history.get(cls, [])
        start_step = total_steps - len(accs)

        padded[cls] = [np.nan] * start_step + accs

    return padded
