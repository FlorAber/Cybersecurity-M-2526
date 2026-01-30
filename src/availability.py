import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import argparse


CLASS_ORDER = ["Brute Force", "DDoS", "DoS", "Injection", "Normal", "Scanning", "XSS"]
NUM_CLASSES = len(CLASS_ORDER)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}

LABEL_MAP = {
    "Brute Force": "Brute Force",

    "DDoS": "DDoS",

    "DoS": "DoS",

    "Injection": "Injection",

    "Normal": "Normal",

    "Scanning": "Scanning",

    "XSS": "XSS",
}

def map_and_filter_labels(X, y, label_encoder, label_map):
    """
    Mappa le label originali CICIDS in macro-classi.
    NON usa il label_encoder per trasformare le macro-classi.
    """

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
    assert len(X) == len(y), 

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

def prepare_data_once(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    label_encoder,
    label_map,
):
    # Mapping
    X_train, y_train = map_and_filter_labels(X_train, y_train, label_encoder, label_map)
    X_val,   y_val   = map_and_filter_labels(X_val, y_val, label_encoder, label_map)
    X_test,  y_test  = map_and_filter_labels(X_test, y_test, label_encoder, label_map)

    # Split per classe (una sola volta)
    train_by_class = split_by_class(X_train, y_train)
    val_by_class   = split_by_class(X_val, y_val)
    test_by_class  = split_by_class(X_test, y_test)

    return train_by_class, val_by_class, test_by_class

def build_step_data(data_by_class, step_classes):
    
    # Costruisce X, y per uno step (unendo più classi)

    X_list, y_list = [], []

    for cls in step_classes:
        assert cls in data_by_class, f"Classe {cls} mancante nei dati"
        Xc, yc = data_by_class[cls]
        X_list.append(Xc)
        y_list.append(yc)

    X = np.vstack(X_list)
    y = np.hstack(y_list) 

    # converte label string in indice
    y = np.array([CLASS_TO_IDX[label] for label in y])
    
    print("Step classes:", step_classes)
    print("X shape:", X.shape)
    print("y distribution:", np.unique(y, return_counts=True))
    
    return X, y

def build_step_cache(data_by_class, class_order):
    step_cache = {}

    for step in range(1, len(class_order) + 1):
        step_classes = class_order[:step]
        step_cache[step] = build_step_data(data_by_class, step_classes)

    return step_cache

def compute_class_weights(y, num_classes):
    """
    Calcola pesi inversamente proporzionali
    alla frequenza delle classi nello step corrente.
    """
    counts = np.bincount(y, minlength=num_classes)
    weights = 1.0 / np.power(counts + 1e-6, 0.3)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch_batch(model, X, y, optimizer, criterion, batch_size=2048):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()


def poison_data_inplace(
    X, y, step,
    poison_fraction,
    noise_level,
    alpha=0.25,
    mask=None,
    seed=42
):
    rng = np.random.default_rng(seed)

    # classi vecchie
    old_idx = np.where(y < step - 1)[0]
    # nuova classe
    new_idx = np.where(y == step - 1)[0]

    if len(old_idx) == 0 or len(new_idx) == 0:
        return X

    n = int(poison_fraction * len(old_idx))
    if n == 0:
        return X

    poison_idx = rng.choice(old_idx, size=n, replace=False)

    if mask is None:
        mask = np.ones(X.shape[1], dtype=bool)

    # 1️ centroide della nuova classe (solo feature selezionate)
    mu_new = X[new_idx][:, mask].mean(axis=0)

    # 2️ spostamento direzionale
    X_poison = X[poison_idx][:, mask]
    X_poison += alpha * (mu_new - X_poison)

    # 3️ rumore residuo (opzionale ma utile)
    if noise_level > 0:
        X_poison += rng.normal(
            0, noise_level, size=X_poison.shape
        )

    X[poison_idx[:, None], mask] = X_poison

    return X


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
        
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(logits, 1).numpy()
    return accuracy_score(y, preds)
    
def train_offline_model(
    X_tr, y_tr,
    X_v, y_v,
    input_dim,
    num_classes,
    lr=1e-3,
    epochs=50,
    patience=5
):
    model = IDSModel(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = -np.inf
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        train_epoch_batch(model, X_tr, y_tr, optimizer, criterion)

        val_acc = evaluate(model, X_v, y_v)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, optimizer


def load_pretrained_model(input_dim, num_classes, name_pretrained_model, lr=1e-3):
    model = IDSModel(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint = torch.load(name_pretrained_model)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return model, optimizer

def run_experiment_fast_pretrained(
    train_steps,
    val_steps,
    test_steps,
    train_by_class,
    strategy="full",
    poisoned=False,
    poison_fraction=0.2,
    noise_level=0.2,
    memory_size=400,
    epochs=10,
    pretrained=False,
    name_model="C:/Users/ASUS/CyberSec Project/pretrained_ids.pt"
):
    history = {c: [] for c in CLASS_ORDER}

    if pretrained:
        model, optimizer = load_pretrained_model(
            input_dim=next(iter(train_steps.values()))[0].shape[1],
            num_classes=NUM_CLASSES,
            name_pretrained_model=name_model
        )
    else:
        model = None
        optimizer = None

    criterion = nn.CrossEntropyLoss()

    for step in range(1, NUM_CLASSES + 1):
        print(f"\n=== STEP {step} ===")

        X_tr, y_tr = train_steps[step]
        X_v,  y_v  = val_steps[step]
        X_te, y_te = test_steps[step]

        # === Strategia di apprendimento ===
        if strategy == "none":
            new_class = CLASS_ORDER[step - 1]
            mask = y_tr == CLASS_TO_IDX[new_class]
            X_tr, y_tr = X_tr[mask], y_tr[mask]

        elif strategy == "memory" and step > 1:
            old_classes = CLASS_ORDER[:step - 1]
            X_old, y_old = sample_memory(train_by_class, old_classes, memory_size)
            X_new, y_new = train_steps[step]
            X_tr = np.vstack([X_old, X_new])
            y_tr = np.hstack([y_old, y_new])

        # Poisoning
        if poisoned and step > 1:
            X_tr = X_tr.copy()
            poison_data_inplace(
                X_tr, y_tr,
                step,
                poison_fraction,
                noise_level
            )

        # Init se non pretrained
        if model is None:
            model = IDSModel(X_tr.shape[1], NUM_CLASSES)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training con early stopping
        best_val = -np.inf
        best_state = None
        patience = 3
        no_improve = 0

        for _ in range(epochs):
            train_epoch_batch(model, X_tr, y_tr, optimizer, criterion)

            val_acc = evaluate(model, X_v, y_v)

            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Test per classe
        for cls_idx, cls_name in enumerate(CLASS_ORDER[:step]):
            mask = y_te == cls_idx
            acc = (
                evaluate(model, X_te[mask], y_te[mask])
                if mask.any()
                else np.nan
            )
            history[cls_name].append(acc)

    return history

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

def last_valid(acc_list):
    acc = np.array(acc_list, dtype=float)
    acc = acc[~np.isnan(acc)]
    return acc[-1] if len(acc) > 0 else np.nan

def load_data(base_path):
    X_train = np.load(os.path.join(base_path, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    X_val   = np.load(os.path.join(base_path, "X_val.npy")).astype(np.float32)
    y_val   = np.load(os.path.join(base_path, "y_val.npy"))
    X_test  = np.load(os.path.join(base_path, "X_test.npy")).astype(np.float32)
    y_test  = np.load(os.path.join(base_path, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test

DATASET_PATHS = {
    "balanced": "C:/Users/ASUS/CyberSec Project/processed_data/processed_data_us_new",
    "unbalanced": "C:/Users/ASUS/CyberSec Project/processed_data/processed_data_fix"
}

def main():

    parser = argparse.ArgumentParser(description="Run availability attack experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["balanced", "unbalanced"],
        default="balanced",
        required=True,
        help="Dataset da utilizzare: 'balanced' o 'unbalanced'."
    )

    args = parser.parse_args()
    dataset_choice = DATASET_PATHS[args.dataset]

    print(f"Using dataset: {dataset_choice}")

    with open(os.path.join(dataset_choice, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_choice)

    # PREPARAZIONE (una volta)
    train_by_class, val_by_class, test_by_class = prepare_data_once(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        label_encoder,
        LABEL_MAP
    )

    train_steps = build_step_cache(train_by_class, CLASS_ORDER)
    val_steps   = build_step_cache(val_by_class, CLASS_ORDER)
    test_steps  = build_step_cache(test_by_class, CLASS_ORDER)

    poison_fractions = [0.1, 0.5, 0.7, 0.9, 1]
    poisoned_results = {}

    for pf in poison_fractions:
        print(f"Poison fraction: {pf}")
        poisoned_results[pf] = run_experiment_fast_pretrained(
            train_steps,
            val_steps,
            test_steps,
            train_by_class,
            strategy="memory",
            poisoned=True,
            poison_fraction=pf,
            pretrained=True,
        )

    baseline_results = run_experiment_fast_pretrained(
        train_steps,
        val_steps,
        test_steps,
        train_by_class,
        strategy="memory",
        poisoned=False,
        pretrained=True,
    )

    # Padding
    baseline_padded = pad_history(baseline_results, CLASS_ORDER, NUM_CLASSES)
    poisoned_padded = {
        pf: pad_history(poisoned_results[pf], CLASS_ORDER, NUM_CLASSES)
        for pf in poison_fractions
    }

    # Heatmap
    classes = list(baseline_padded.keys())
    heatmap_data = np.zeros((len(classes), len(poison_fractions)))

    for i, cls in enumerate(classes):
        base_final = last_valid(baseline_padded[cls])
        for j, pf in enumerate(poison_fractions):
            heatmap_data[i, j] = base_final - last_valid(poisoned_padded[pf][cls])

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        heatmap_data,
        xticklabels=poison_fractions,
        yticklabels=classes,
        annot=True,
        fmt=".3f",
        cmap="Reds"
    )
    plt.xlabel("Poison fraction")
    plt.ylabel("Class")
    plt.title("Accuracy degradation (baseline − poisoned)")
    plt.tight_layout()
    plt.show()



    
if __name__ == "__main__":
    main()
