import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from collections import defaultdict
import matplotlib.pyplot as plt  # type: ignore
import pickle

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

seed = 42
np.random.seed(seed)

SOURCE_PATH = "../data/processed_data_fix/"
PRETRAINED_MODEL_PATH = "./checkpoints/ids_checkpoint.pt"
USE_PRETRAINED = True
NUM_STEPS = 10
POISONING_RATE = 0.1


# Definizione modello di classificazione (Rete Neurale MLP)
class IDSModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Salvataggio modello su file system
def save_model(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    label_to_idx: Optional[Dict[Any, int]] = None,
    idx_to_label: Optional[list] = None,
    input_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
) -> str:
    p = Path(path)
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
        p = p / "checkpoint.pt"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)

    if input_dim is None:
        input_dim = getattr(getattr(model, "net", None)[0], "in_features", None)
    if num_classes is None:
        num_classes = getattr(getattr(model, "net", None)[-1], "out_features", None)

    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "label_to_idx": label_to_idx if label_to_idx is not None else {},
        "idx_to_label": idx_to_label if idx_to_label is not None else [],
        "extra": {
            "input_dim": int(input_dim) if input_dim is not None else None,
            "num_classes": int(num_classes) if num_classes is not None else None,
        },
    }

    torch.save(ckpt, str(p))
    return str(p)


# Caricamento modello da file system, restituisce model, optimizer, label_to_idx, idx_to_label, extra
def load_model(
    path: str,
    model_cls=IDSModel,
    device: str = "cpu",
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.nn.Module,
    Optional[torch.optim.Optimizer],
    Dict[Any, int],
    list,
    Dict[str, Any],
]:

    ckpt = torch.load(path, map_location=device, weights_only=False)

    extra = ckpt.get("extra", {})
    input_dim = extra.get("input_dim")
    num_classes = extra.get("num_classes")
    if input_dim is None or num_classes is None:
        raise ValueError("Checkpoint privo di input_dim/num_classes in ckpt['extra'].")

    model = model_cls(input_dim, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    optimizer = None
    opt_state = ckpt.get("optimizer_state", None)
    if opt_state is not None:
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-3}
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        optimizer.load_state_dict(opt_state)

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    label_to_idx = ckpt.get("label_to_idx", {})
    idx_to_label = ckpt.get("idx_to_label", [])

    return model, optimizer, label_to_idx, idx_to_label, extra


# Caricamento dataset splitted (train,test,val,scaler,label_encoder,feature_names)
def load_splitted_dataset(path: str = SOURCE_PATH):
    X_train = np.load(SOURCE_PATH + "X_train.npy")
    y_train = np.load(SOURCE_PATH + "y_train.npy")

    X_test = np.load(SOURCE_PATH + "X_test.npy")
    y_test = np.load(SOURCE_PATH + "y_test.npy")

    X_val = np.load(SOURCE_PATH + "X_val.npy")
    y_val = np.load(SOURCE_PATH + "y_val.npy")

    with open(SOURCE_PATH + "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(SOURCE_PATH + "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open(SOURCE_PATH + "feature_names.pkl", "rb") as f:
        feature_names = np.array(pickle.load(f))

    print(
        f"\nLoaded datasets with shapes:\n "
        f" Train : {X_train.shape}\n"
        f" Test : {X_test.shape}\n"
        f" Val : {X_val.shape}"
    )
    print(
        f"Loaded pickle objects:\n"
        f" Scaler: {type(scaler)}\n"
        f" Label encoder: {type(label_encoder)}\n"
        f" Feature array: {feature_names}\n"
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        scaler,
        label_encoder,
        feature_names,
    )


# Decodifica delle labels
def decode_labels(y: np.ndarray, label_encoder):
    if np.issubdtype(y.dtype, np.integer):
        assert label_encoder is not None
        y = label_encoder.inverse_transform(y)

    return y


# Estrae lista delle classi presenti nel dataset (con decodifica se necessario)
def extract_labels(y: np.ndarray, label_encoder):
    if np.issubdtype(y.dtype, np.integer):
        assert label_encoder is not None
        y = label_encoder.inverse_transform(y)

    return np.unique(y)


# Riordina le classi casualmente (se viene passata order le classi vengono prese in quell'ordine e quelle mancanti vengono aggiunte in coda)
def random_order_classes(y, label_encoder, order=[]):
    classes = extract_labels(y, label_encoder)
    ordered = []

    if len(order) > 0:
        for c in order:
            idx = np.where(classes == c)[0]
            if idx.size <= 0:
                print(f"Classe {c} non presente in y, ignoro")
            else:
                ordered.append(c)
                classes = np.delete(classes, idx[0])

    np.random.default_rng(seed).shuffle(classes)
    ordered.extend(classes)

    return ordered


# Divide X e y per classe, restituisce un dizionario del tipo: classe -> (X_class,y_class)
def split_by_class(X, y):
    assert len(X) == len(y), "X e y hanno lunghezze diverse"

    data = defaultdict(list)

    for xi, yi in zip(X, y):
        data[yi].append(xi)

    split_data = {}
    for cls, samples in data.items():
        X_cls = np.array(samples)
        y_cls = np.array([cls] * len(samples))
        split_data[cls] = (X_cls, y_cls)

    for cls, (Xc, _) in split_data.items():
        print(f"{cls:<12} : {len(Xc):>7} samples")

    return split_data


# Decodifica le labels e divide i dataset per classe, restituisce tre dizionari
def prepare_data(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    y_train = decode_labels(y_train, label_encoder)
    y_val = decode_labels(y_val, label_encoder)
    y_test = decode_labels(y_test, label_encoder)

    print("\nTrain: ")
    train_by_class = split_by_class(X_train, y_train)
    print("\nValidation: ")
    val_by_class = split_by_class(X_val, y_val)
    print("\nTest: ")
    test_by_class = split_by_class(X_test, y_test)

    return train_by_class, val_by_class, test_by_class


# Costruisce X e y per uno step unendo più classi, preleva dal dizionario data_by_class le classi presenti nella lista step_classes
# y viene codificato secondo l'encoder precedentemente caricato
def build_step_data(data_by_class, class_order, step_classes, class_to_idx=None):
    if class_to_idx is None:
        class_to_idx = {c: i for i, c in enumerate(class_order)}

    X_list, y_list = [], []

    for cls in step_classes:
        assert cls in data_by_class, f"Classe {cls} mancante nei dati"
        Xc, yc = data_by_class[cls]
        X_list.append(Xc)
        y_list.append(yc)

    X = np.vstack(X_list)
    y_str = np.hstack(y_list)

    y = np.array([class_to_idx[label] for label in y_str], dtype=np.int64)

    print(" Step classes:", ", ".join(map(str, step_classes)))
    print("  X shape : ", X.shape)
    print(
        "  y distribution: ["
        + ", ".join(f"{c}:{n}" for c, n in zip(*np.unique(y, return_counts=True)))
        + "]"
    )

    return X, y


# Costruisce il dataset suddiviso per step dividendo i campioni di ogni classe in chunk
# Ogni step contiene tutte le classi, con campioni stratificati per num_steps
def build_step_cache(data_by_class, class_order, num_steps=None):
    n_classes = len(class_order)
    if num_steps is None:
        num_steps = 1
    if num_steps <= 0:
        raise ValueError("num_steps deve essere >= 1")

    class_to_idx = {c: i for i, c in enumerate(class_order)}

    per_class_chunks = {}
    for cls in class_order:
        assert cls in data_by_class, f"Classe {cls} mancante nei dati"
        Xc, yc = data_by_class[cls]

        idx = np.random.permutation(len(Xc))
        Xc = Xc[idx]
        yc = yc[idx]

        X_splits = np.array_split(Xc, num_steps)
        y_splits = np.array_split(yc, num_steps)
        per_class_chunks[cls] = list(zip(X_splits, y_splits))

        if len(Xc) < num_steps:
            print(
                f"[WARN] Classe '{cls}' ha {len(Xc)} campioni < num_steps={num_steps}: "
                "alcuni step potrebbero non contenere questa classe."
            )

    step_cache = {}

    for step in range(1, num_steps + 1):
        X_list, y_list = [], []
        for cls in class_order:
            X_part, y_part = per_class_chunks[cls][step - 1]
            if len(X_part) == 0:
                continue
            X_list.append(X_part)
            y_list.append(y_part)

        if len(X_list) == 0:
            X = np.empty((0, 0), dtype=np.float32)
            y = np.empty((0,), dtype=np.int64)
        else:
            X = np.vstack(X_list)
            y_str = np.hstack(y_list)
            y = np.array([class_to_idx[label] for label in y_str], dtype=np.int64)

        print(f"\nSTEP {step}")
        print(" Step classes:", ", ".join(map(str, class_order)))
        print("  X shape : ", X.shape)
        if len(y) > 0:
            dist = ", ".join(
                f"{c}:{n}" for c, n in zip(*np.unique(y, return_counts=True))
            )
        else:
            dist = ""
        print(f"  y distribution: [{dist}]")
        step_cache[step] = (X, y)

    return step_cache


# Allena il modello per un'epoca dividendo i dati in batch
def train_epoch_batch(model, X, y, optimizer, criterion, batch_size=2048):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()


# Campiona al massimo memory_size entry per ciascuna classe, restituisce X e y con label numeriche
def sample_memory(data_by_class, classes, class_order, memory_size):
    X_list, y_list = [], []

    for cls in classes:
        Xc, yc = data_by_class[cls]
        n = min(memory_size, len(Xc))
        idx = np.random.choice(len(Xc), size=n, replace=False)

        X_list.append(Xc[idx])
        y_list.append(
            np.full(n, {c: i for i, c in enumerate(class_order)}[cls], dtype=np.int64)
        )

    if len(X_list) == 0:
        return None, None

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y


# Applica targeted data poisoning con label flipping ad una percentuale di campioni della classe target
# I campioni vengono riassegnati alla benign_class_idx o a classi casuali se benign_class_idx=None
def apply_targeted_poisoning(
    X, y, target_class_idx, class_order, poisoning_rate, benign_class_idx
):
    if poisoning_rate <= 0.0 or poisoning_rate > 1.0:
        return X, y

    target_mask = y == target_class_idx
    target_indices = np.where(target_mask)[0]

    if len(target_indices) == 0:
        return X, y

    num_to_poison = int(len(target_indices) * poisoning_rate)

    if num_to_poison == 0:
        return X, y

    poison_indices = np.random.choice(target_indices, size=num_to_poison, replace=False)

    y_poisoned = y.copy()
    if benign_class_idx is not None:
        y_poisoned[poison_indices] = benign_class_idx
    else:
        available_classes = [
            i for i in range(len(class_order)) if i != target_class_idx
        ]

        if len(available_classes) == 0:
            return X, y

        for idx in poison_indices:
            new_label = np.random.choice(available_classes)
            y_poisoned[idx] = new_label

    print(
        f"  → Poisoned {num_to_poison}/{len(target_indices)} samples of class '{class_order[target_class_idx]}' (rate: {poisoning_rate:.1%})"
    )

    return X, y_poisoned


# Funzione di valutazione del modello su X e y, restituisce l'accuracy score
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(logits, 1).numpy()
    return accuracy_score(y, preds)


# Espansione head del modello per aggiunta nuove classi trovate
def expand_head(model, new_num_classes):
    old_head = model.net[4]
    assert isinstance(old_head, nn.Linear)

    old_num = old_head.out_features
    if new_num_classes <= old_num:
        return model

    new_head = nn.Linear(old_head.in_features, new_num_classes)

    with torch.no_grad():
        new_head.weight[:old_num].copy_(old_head.weight)
        new_head.bias[:old_num].copy_(old_head.bias)

    model.net[4] = new_head
    return model


# Esecuzione dell'incremental learning per step con poisoning
#   Per ogni step:
#   1) Caricamento dati di train/val/test per lo step corrente
#   2) Aggiornamento mapping delle label originali in indici contigui e inizializzazione history per eventuali nuove classi
#   3) SE strategia "memory" campiona alcuni esempi delle classi già viste e le concatena ai dati di train correnti per ridurre il forgetting
#   4) SE poisoning_rate > 0, applica targeted poisoning alle classi vecchie (già viste negli step precedenti) dal secondo step in poi
#   5) Inizializzazione del modello (se primo step) o espansione dell'head in presenza di nuove classi, ricreazione optimizer
#   6) Conversione liste di label in etichette contigue secondo il mapping interno
#   7) Addestramento per un massimo di 'epochs' con early stopping su validation, salva i pesi migliori
#   8) Valutazione con test e salvataggio in history dell'accuracy
# Restituisce: dict {label: [accuracy_per_step,...]}, model, optimizer, label_to_idx, idx_to_label
def run_training(
    train_steps,
    val_steps,
    test_steps,
    train_by_class,
    class_order,
    memory_size=200,
    epochs=10,
    lr=1e-3,
    patience=3,
    model=None,
    poisoning_rate=0.0,
    pretrained_label_to_idx=None,
    pretrained_idx_to_label=None,
):
    history = defaultdict(list)

    optimizer = None
    criterion = nn.CrossEntropyLoss()

    if pretrained_label_to_idx is not None and pretrained_idx_to_label is not None:
        label_to_idx = pretrained_label_to_idx.copy()
        idx_to_label = pretrained_idx_to_label.copy()
        print(f"\nUsing pretrained model mappings:")
        print(f"  Classes already known: {idx_to_label}")
    else:
        label_to_idx = {}
        idx_to_label = []

    def map_labels(y):
        return np.array([label_to_idx[l] for l in y], dtype=np.int64)

    for step in sorted(train_steps.keys()):
        print(f"\n=== STEP {step} ===")
        X_tr, y_tr = train_steps[step]
        X_v, y_v = val_steps[step]
        X_te, y_te = test_steps[step]

        y_tr_str = np.array([class_order[int(idx)] for idx in y_tr])
        y_v_str = np.array([class_order[int(idx)] for idx in y_v])
        y_te_str = np.array([class_order[int(idx)] for idx in y_te])

        prev_steps = len(next(iter(history.values()))) if history else 0

        new_classes_this_step = []
        for lab in np.unique(y_tr_str):
            if lab not in label_to_idx:
                new_classes_this_step.append(lab)
                label_to_idx[lab] = len(idx_to_label)
                idx_to_label.append(lab)

                history[lab] = [np.nan] * prev_steps

        num_seen = len(idx_to_label)

        if num_seen > 1:
            old_label_names = idx_to_label[:-1]

            X_old, y_old = sample_memory(
                train_by_class, old_label_names, class_order, memory_size
            )

            if X_old is not None:
                y_old_str = np.array([class_order[int(idx)] for idx in y_old])

                X_tr = np.vstack([X_old, X_tr])
                y_tr_str = np.hstack([y_old_str, y_tr_str])

        if step > 1:
            y_tr_numeric = np.array([class_order.index(lab) for lab in y_tr_str])

            old_classes_this_step = [
                lab for lab in np.unique(y_tr_str) if lab not in new_classes_this_step
            ]

            if len(old_classes_this_step) > 0:
                print(f"Applying poisoning to old classes: {old_classes_this_step}")
                for old_class in old_classes_this_step:
                    old_class_idx = class_order.index(old_class)
                    benign_class_idx = class_order.index("Normal")

                    X_tr, y_tr_numeric = apply_targeted_poisoning(
                        X_tr,
                        y_tr_numeric,
                        old_class_idx,
                        class_order,
                        poisoning_rate * step,
                        benign_class_idx,
                    )

                y_tr_str = np.array([class_order[int(idx)] for idx in y_tr_numeric])

        if model is None:
            model = IDSModel(X_tr.shape[1], num_seen)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            model = expand_head(model, num_seen)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        y_tr_m = map_labels(y_tr_str)
        y_v_m = map_labels(y_v_str)
        y_te_m = map_labels(y_te_str)

        best_val = -np.inf
        best_state = None
        no_improve = 0

        for _ in range(epochs):
            train_epoch_batch(model, X_tr, y_tr_m, optimizer, criterion)
            val_acc = evaluate(model, X_v, y_v_m)

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

        for lab in idx_to_label:
            mask = y_te_str == lab
            acc = (
                evaluate(model, X_te[mask], map_labels(y_te_str[mask]))
                if mask.any()
                else np.nan
            )
            history[lab].append(acc)

    return dict(history), model, optimizer, label_to_idx, idx_to_label


if __name__ == "__main__":

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        scaler,
        label_encoder,
        feature_names,
    ) = load_splitted_dataset(SOURCE_PATH)

    class_order = random_order_classes(y_train, label_encoder, order=["Normal", "DDoS"])
    print(f"\nClass discovery order: {class_order}")

    train_by_class, val_by_class, test_by_class = prepare_data(
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
    )

    print("\nCostruzione step cache per TRAIN")
    train_steps = build_step_cache(
        train_by_class, class_order=class_order, num_steps=NUM_STEPS
    )
    print("\nCostruzione step cache per VALIDATION")
    val_steps = build_step_cache(
        val_by_class, class_order=class_order, num_steps=NUM_STEPS
    )
    print("\nCostruzione step cache per TEST")
    test_steps = build_step_cache(
        test_by_class, class_order=class_order, num_steps=NUM_STEPS
    )

    pretrained_model = None
    pretrained_label_to_idx = None
    pretrained_idx_to_label = None

    if USE_PRETRAINED:

        try:
            (
                pretrained_model,
                _,
                pretrained_label_to_idx,
                pretrained_idx_to_label,
                extra,
            ) = load_model(PRETRAINED_MODEL_PATH)
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Starting from scratch instead")
            pretrained_model = None
            pretrained_label_to_idx = None
            pretrained_idx_to_label = None

    history, model, optimizer, label_to_idx, idx_to_label = run_training(
        train_steps=train_steps,
        val_steps=val_steps,
        test_steps=test_steps,
        train_by_class=train_by_class,
        class_order=class_order,
        epochs=10,
        poisoning_rate=POISONING_RATE,
        model=pretrained_model,
        pretrained_label_to_idx=pretrained_label_to_idx,
        pretrained_idx_to_label=pretrained_idx_to_label,
    )

    plt.figure(figsize=(10, 5))

    plotted = set()
    for idx, name in enumerate(class_order):
        if idx in history:
            accs = history[idx]
            y = np.asarray(accs, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=name)
            plotted.add(idx)

    for cls, accs in history.items():
        if cls in plotted:
            continue
        y = np.asarray(accs, dtype=float)
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=str(cls))

    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.legend(title="Class", loc="best")
    plt.tight_layout()
    plt.show()

    plt.savefig("./trainpois-pre-full-01-10.png")

    # ckpt_path = save_model(
    #     path=MOD_PATH,
    #     model=model,
    #     optimizer=optimizer,
    #     label_to_idx=label_to_idx,
    #     idx_to_label=idx_to_label,
    #     input_dim=X_train.shape[1],
    #     num_classes=model.net[-1].out_features,
    # )

    # print("Saved checkpoint at:", ckpt_path)
