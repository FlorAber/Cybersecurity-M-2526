import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle
from utils.comp_result import *

# VARIABILI
LABEL_MAP = {
    "BENIGN": "Normal",

    "DDoS": "DoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",

    "PortScan": "Probe",

    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
}



CLASS_ORDER = ["Normal", "DoS", "Probe", "BruteForce"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)
# DATA UTILS
def map_and_filter_labels(X, y, label_encoder, label_map):
    """
    Mappa le label originali CICIDS in macro-classi.
    NON usa il label_encoder per trasformare le macro-classi.
    """

    # 1. se y è numerico → torniamo alle label originali
    if np.issubdtype(y.dtype, np.integer):
        assert label_encoder is not None, "label_encoder richiesto per y numerico"
        y_str = label_encoder.inverse_transform(y)
    else:
        y_str = y  # già stringhe

    X_out = []
    y_out = []

    for xi, yi in zip(X, y_str):
        if yi in label_map:
            X_out.append(xi)
            y_out.append(label_map[yi])
            
    print(f"Kept {len(y_out)} / {len(y)} samples after label filtering")
    return np.array(X_out), np.array(y_out)

def split_by_class(X, y):
    """
    Divide X e y per macro-classe.

    Parametri
    ----------
    X : np.ndarray (N, F)
        Feature matrix (già preprocessata / scalata)
    y : np.ndarray (N,)
        Label come STRINGHE (Normal, DoS, Probe, BruteForce)

    Ritorna
    -------
    dict :
        macro_classe -> (X_class, y_class)
    """
    assert len(X) == len(y), "X e y hanno lunghezze diverse"

    data = defaultdict(list)

    # raggruppa i campioni per classe
    for xi, yi in zip(X, y):
        data[yi].append(xi)

    # conversione in numpy array
    split_data = {}
    for cls, samples in data.items():
        X_cls = np.array(samples)
        y_cls = np.array([cls] * len(samples))
        split_data[cls] = (X_cls, y_cls)
        
    for cls, (Xc, _) in split_data.items():
        print(f"{cls}: {len(Xc)} samples")


    return split_data

def build_step_data(data_by_class, step_classes):
    
    # Costruisce X, y per uno step (unendo più classi)

    X_list, y_list = [], []

    for cls in step_classes:
        assert cls in data_by_class, f"Classe {cls} mancante nei dati"
        Xc, yc = data_by_class[cls]
        X_list.append(Xc)
        y_list.append(yc)

    X = np.vstack(X_list) # concatena le righe
    y = np.hstack(y_list) # concatena i vettori

    # converte label string → indice
    y = np.array([CLASS_TO_IDX[label] for label in y])
    
    print("Step classes:", step_classes)
    print("X shape:", X.shape)
    print("y distribution:", np.unique(y, return_counts=True))
    
    return X, y

'''
split_by_class
→ separa il dataset per classe (statica)

step_classes
→ definisce quali classi sono visibili allo step

build_step_data
→ materializza lo step:

crea il dataset reale

con SOLO le classi consentite

mappa le label nel formato giusto
'''

def normalize_step(X_train, X_val, X_test):
    '''
    I dati sono stati già stati scalati nel preprocessing
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n   = scaler.transform(X_val)
    X_test_n  = scaler.transform(X_test)
    return X_train_n, X_val_n, X_test_n
    '''
    return X_train, X_val, X_test

def balance_dataset_by_min_class(X, y, random_state=42):
    """
    Bilancia il dataset prendendo per ogni classe
    lo stesso numero di campioni, pari alla classe meno rappresentata.

    Parametri
    ----------
    X : np.ndarray (N, F)
    y : np.ndarray (N,)  # STRINGHE
    random_state : int

    Ritorna
    -------
    X_bal, y_bal
    """
    np.random.seed(random_state)

    # raggruppa per classe
    class_indices = defaultdict(list)
    for idx, cls in enumerate(y):
        class_indices[cls].append(idx)

    # numero minimo di campioni tra le classi
    min_count = min(len(idxs) for idxs in class_indices.values())

    # campionamento bilanciato
    selected_indices = []
    for cls, idxs in class_indices.items():
        selected = np.random.choice(idxs, size=min_count, replace=False)
        selected_indices.extend(selected)

    # shuffle finale
    selected_indices = np.random.permutation(selected_indices)

    return X[selected_indices], y[selected_indices]


# Funzione di supporto per il Limited Memory Replay, 
# Questa funzione campiona K esempi per classe vecchia.
def sample_memory(data_by_class, classes, memory_size):
    """
    Campiona al massimo memory_size esempi per ciascuna classe.
    Ritorna X, y con label NUMERICHE.
    """
    X_list, y_list = [], []

    for cls in classes:
        Xc, yc = data_by_class[cls]
        n = min(memory_size, len(Xc))
        idx = np.random.choice(len(Xc), size=n, replace=False)

        X_list.append(Xc[idx])
        y_list.append(
            np.full(n, CLASS_TO_IDX[cls], dtype=np.int64)
        )

    if len(X_list) == 0:
        return None, None

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

def compute_class_weights(y, num_classes):
    """
    Calcola pesi inversamente proporzionali
    alla frequenza delle classi nello step corrente.
    """
    counts = np.bincount(y, minlength=num_classes)
    weights = 1.0 / np.power(counts + 1e-6, 0.3)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
################################################################
# MODEL
class IDSModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
#################################################################
# POISONING
def poison_data_realistic(
    X,
    y,
    current_step,
    noise_level=0.1,
    poison_fraction=0.2,
    sensitive_feature_mask=None,
    random_state=None
):
    """
    Data poisoning realistico per dataset grandi e continual learning.

    Parametri
    ----------
    X : np.ndarray (N, F)
        Dati NORMALIZZATI
    y : np.ndarray (N,)
        Label come indici (0,1,2,...)
    current_step : int
        Step corrente (1-based)
    noise_level : float
        Intensità del rumore
    poison_fraction : float
        Percentuale di campioni avvelenati
    sensitive_feature_mask : np.ndarray (F,) o None
        True per feature avvelenabili
    random_state : int o None
        Seed

    Ritorna
    -------
    X_poisoned : np.ndarray
    """
    if random_state is not None:
        np.random.seed(random_state)

    Xp = X.copy()

    # classi vecchie
    old_mask = y < (current_step - 1)
    old_indices = np.where(old_mask)[0]

    if len(old_indices) == 0:
        return Xp

    num_poison = int(poison_fraction * len(old_indices))
    poison_indices = np.random.choice(
        old_indices, size=num_poison, replace=False
    )

    if sensitive_feature_mask is None:
        sensitive_feature_mask = np.ones(X.shape[1], dtype=bool)

    noise = np.random.normal(
        0.0,
        noise_level,
        size=(num_poison, sensitive_feature_mask.sum())
    )

    Xp[poison_indices][:, sensitive_feature_mask] += noise
    return Xp
#################################################################
# TRAINING
def train_epoch(model, X, y, optimizer, criterion):
    model.train()
    
    if len(X) == 0:
        return np.nan
        
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
################################################################
# MAIN FUNCTION
def run_experiment(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    label_encoder,
    poisoned=False,
    balanced=False,
    strategy="full", # full, none, memory
    memory_size=200,
    noise_level=0.1,
    poison_fraction=0.2,
    epochs = 10,
    feature_mask=None,
    label_map=LABEL_MAP
):

    # =========================
    # 1. Mapping macro-classi
    # =========================
    X_train, y_train = map_and_filter_labels(X_train, y_train, label_encoder, label_map)
    X_val,   y_val   = map_and_filter_labels(X_val, y_val, label_encoder, label_map)
    X_test,  y_test  = map_and_filter_labels(X_test, y_test, label_encoder, label_map)

    # =========================
    # 2. Bilanciamento globale (opzionale)
    # =========================
    if balanced:
        print(">> Using BALANCED datasets")
        X_train, y_train = balance_dataset_by_min_class(X_train, y_train)
        X_val,   y_val   = balance_dataset_by_min_class(X_val, y_val)
        X_test,  y_test  = balance_dataset_by_min_class(X_test, y_test)
    else:
        print(">> Using ORIGINAL (imbalanced) datasets")

    # =========================
    # 3. Split per classe
    # =========================
    train_by_class = split_by_class(X_train, y_train)
    val_by_class   = split_by_class(X_val, y_val)
    test_by_class  = split_by_class(X_test, y_test)

    model = None
    history = {c: [] for c in CLASS_ORDER}

    # =========================
    # 4. Class-incremental loop
    # =========================
    for step in range(1, NUM_CLASSES + 1):

        mode = "POISONED" if poisoned else "BASELINE"
        print(f"\n=== STEP {step} {mode} ===")

        step_classes = CLASS_ORDER[:step]

        step_classes = CLASS_ORDER[:step]
        new_class = CLASS_ORDER[step - 1]

        # =========================
        # TRAINING DATA SELECTION
        # =========================
        if strategy == "full":
            # FULL REHEARSAL
            X_tr, y_tr = build_step_data(train_by_class, step_classes)

        elif strategy == "none":
            # NO REHEARSAL
            X_tr, y_tr_str = train_by_class[new_class]
            y_tr = np.full(len(y_tr_str), CLASS_TO_IDX[new_class], dtype=np.int64)

        elif strategy == "memory":
            # LIMITED MEMORY REPLAY
            old_classes = CLASS_ORDER[:step - 1]

            X_old, y_old = sample_memory(
                train_by_class,
                old_classes,
                memory_size
            )

            X_new, y_new = build_step_data(train_by_class, [new_class])

            if X_old is None:
                X_tr, y_tr = X_new, y_new
            else:
                X_tr = np.vstack([X_old, X_new])
                y_tr = np.hstack([y_old, y_new])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

            
        X_v,  y_v  = build_step_data(val_by_class, step_classes)
        X_te, y_te = build_step_data(test_by_class, step_classes)

        # Se i dati sono già scalati globalmente, questa funzione deve essere identity
        # oppure puoi commentarla del tutto
        # X_tr, X_v, X_te = normalize_step(X_tr, X_v, X_te)

        # =========================
        # Poisoning
        # =========================
        if poisoned and step > 1:
            X_tr = poison_data_realistic(
                X_tr, y_tr,
                current_step=step,
                noise_level=noise_level,
                poison_fraction=poison_fraction,
                sensitive_feature_mask=feature_mask,
                random_state=42
            )

        # =========================
        # Init modello
        # =========================
        if model is None:
            model = IDSModel(X_tr.shape[1], NUM_CLASSES)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            class_weights = compute_class_weights(y_tr, NUM_CLASSES)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # =========================
        # Training
        # =========================
        for _ in range(epochs):
            train_epoch(model, X_tr, y_tr, optimizer, criterion)

        # =========================
        # Evaluation per classe
        # =========================
        for cls_idx, cls_name in enumerate(step_classes):
            mask = y_te == cls_idx

            if mask.sum() == 0:
                acc = np.nan
            else:
                acc = evaluate(model, X_te[mask], y_te[mask])

            history[cls_name].append(acc)
            print(f"Accuracy {cls_name}: {acc:.3f}")

    return history
##########################################################################