import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report  # type: ignore
from collections import defaultdict
import matplotlib.pyplot as plt  # type: ignore
import pickle

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

seed = 42
np.random.seed(seed)

# MOD_PATH = "./checkpoint_us_train.pt"
SOURCE_PATH = "../data/processed_data_fix/"  # Path to source files FOLDER
NUM_STEPS = 7


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
    if p.suffix == "":  # path è una directory o non ha estensione
        p.mkdir(parents=True, exist_ok=True)
        p = p / "checkpoint.pt"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)

    # prova a inferire dimensioni se non passate
    if input_dim is None:
        # assume primo Linear nel modello
        input_dim = getattr(getattr(model, "net", None)[0], "in_features", None)
    if num_classes is None:
        # assume ultimo Linear nel modello
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
    model_cls=IDSModel,  # Classe del modello
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

    ckpt = torch.load(path, map_location=device)

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

        # sposta eventuali tensori nello state dell'optimizer sul device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    label_to_idx = ckpt.get("label_to_idx", {})
    idx_to_label = ckpt.get("idx_to_label", [])

    return model, optimizer, label_to_idx, idx_to_label, extra


# Caricamento dataset splitted (train,test,val,scaler,label_encoder,feature_names)
def load_splitted_dataset(path: str = SOURCE_PATH):
    # Loading train dataset
    X_train = np.load(SOURCE_PATH + "X_train.npy")
    y_train = np.load(SOURCE_PATH + "y_train.npy")

    # Loading test dataset
    X_test = np.load(SOURCE_PATH + "X_test.npy")
    y_test = np.load(SOURCE_PATH + "y_test.npy")

    # Loading validation dataset
    X_val = np.load(SOURCE_PATH + "X_val.npy")
    y_val = np.load(SOURCE_PATH + "y_val.npy")

    # Loading pickle object scaler
    with open(SOURCE_PATH + "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Loading pickle object label encoder
    with open(SOURCE_PATH + "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Loading pickle object feature names
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
    for c in classes:
        ordered.append(c)

    return ordered


# Divide X e y per classe, restituisce un dizionario del tipo: classe -> (X_class,y_class)
def split_by_class(X, y):
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
        print(f"{cls:<12} : {len(Xc):>7} samples")

    return split_data


# Decodifica le labels e divide i dataset per classe, restituisce tre dizionari
def prepare_data(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    y_train = decode_labels(y_train, label_encoder)
    y_val = decode_labels(y_val, label_encoder)
    y_test = decode_labels(y_test, label_encoder)

    # Split per classe
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


# Costruisce il dataset suddiviso per step sfruttando la funzione build_step_data
# Se num_steps <= numero di classi, le distribuisce su num_steps; se > dopo l'ingresso dell'ultima classe gli step restanti le conterranno tutte
def build_step_cache(data_by_class, class_order, num_steps=None):
    n_classes = len(class_order)
    if num_steps is None:
        num_steps = n_classes
    if num_steps <= 0:
        raise ValueError("num_steps deve essere >= 1")

    step_cache = {}
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    # Caso numero di step sia minore delle classi (più classi nuove per ogni step)
    if num_steps <= n_classes:
        chunks = np.array_split(class_order, num_steps)

        seen = []
        for step, chunk in enumerate(chunks, start=1):
            seen.extend(list(chunk))
            print(f"\nSTEP {step}")
            step_cache[step] = build_step_data(
                data_by_class, class_order, seen, class_to_idx=class_to_idx
            )
    # Caso numero di step maggiore o uguale al numero di classi (ogni step introduce una nuova classe)
    else:
        for step in range(1, n_classes + 1):
            step_classes = class_order[:step]
            print(f"\nSTEP {step}")
            step_cache[step] = build_step_data(
                data_by_class, class_order, step_classes, class_to_idx=class_to_idx
            )

        # plateau: tutti gli step restanti hanno tutte le classi
        full_data = step_cache[n_classes]  # (X_all, y_all)
        full_str = ", ".join(map(str, class_order))

        for step in range(n_classes + 1, num_steps + 1):
            print(f"\nSTEP {step}")
            print(f"Step classes: {full_str}")
            step_cache[step] = full_data

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


# Funzione di valutazione del modello su X e y, restituisce l'accuracy score
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(logits, 1).numpy()
    return accuracy_score(y, preds)


# Espansione head del modello per aggiunta nuove classi trovate
def expand_head(model, new_num_classes):
    old_head = model.net[4]  # net.4
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


# Esecuzione dell'incremental learning per step
#   Per ogni step:
#   1) Caricamento dati di train/val/test per lo step corrente
#   2) Aggiornamento mapping delle label originali in indici contigui e inizializzazione history per eventuali nuove classi
#   3) SE strategia "memory" campiona alcuni esempi delle classi già viste e le concatena ai dati di train correnti per ridurre il forgetting
#   4) Inizializzazione del modello (se primo step) o espansione dell'head in presenza di nuove classi, ricreazione optimizer
#   5) Conversione liste di label in etichette contigue secondo il mapping interno
#   6) Addestramento per un massimo di 'epochs' con early stopping su validation, salva i pesi migliori
#   7) Valutazione con test e salvataggio in history dell'accuracy
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
):
    history = defaultdict(list)

    optimizer = None
    criterion = nn.CrossEntropyLoss()

    label_to_idx = {}
    idx_to_label = []

    def map_labels(y):
        # y contiene label originali (anche non contigue); ritorna indici interni contigui
        return np.array([label_to_idx[l] for l in y], dtype=np.int64)

    for step in sorted(train_steps.keys()):
        print(f"\n=== STEP {step} ===")
        # 1) Caricamento dati
        X_tr, y_tr = train_steps[step]
        X_v, y_v = val_steps[step]
        X_te, y_te = test_steps[step]

        # NOTA: y_tr, y_v, y_te contengono già indici numerici (0,1,2...) basati su class_order
        # perché build_step_data li ha convertiti. Dobbiamo decodificarli in label originali.
        y_tr_str = np.array([class_order[int(idx)] for idx in y_tr])
        y_v_str = np.array([class_order[int(idx)] for idx in y_v])
        y_te_str = np.array([class_order[int(idx)] for idx in y_te])

        # 2) Aggiornamento mapping con nuove classi comparse nello step
        prev_steps = len(next(iter(history.values()))) if history else 0

        for lab in np.unique(y_tr_str):
            if lab not in label_to_idx:
                label_to_idx[lab] = len(idx_to_label)
                idx_to_label.append(lab)

                # nuova classe: NaN per tutti gli step precedenti
                history[lab] = [np.nan] * prev_steps

        num_seen = len(idx_to_label)

        # 3) Strategia "memory" - campiona esempi delle classi già viste
        if num_seen > 1:
            # idx_to_label contiene i nomi delle classi (stringhe)
            # Escludiamo l'ultima classe (quella appena aggiunta in questo step)
            old_label_names = idx_to_label[:-1]

            X_old, y_old = sample_memory(
                train_by_class, old_label_names, class_order, memory_size
            )

            if X_old is not None:
                # sample_memory restituisce y_old come indici numerici basati su class_order
                # Dobbiamo convertirli nelle label originali (stringhe)
                y_old_str = np.array([class_order[int(idx)] for idx in y_old])

                X_tr = np.vstack([X_old, X_tr])
                y_tr_str = np.hstack([y_old_str, y_tr_str])

        # 4) Inizializzazione o espansione del modello
        if model is None:
            model = IDSModel(X_tr.shape[1], num_seen)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            # Espandi la testa se sono apparse nuove classi
            model = expand_head(model, num_seen)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 5) Mappatura delle label per training/val/test
        y_tr_m = map_labels(y_tr_str)
        y_v_m = map_labels(y_v_str)
        y_te_m = map_labels(y_te_str)

        # 6) Training con early stopping
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

        # 7) Test per classe (label originali)
        # Assicura che tutte le classi viste appendano un valore questo step
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

    # Loading datasets
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

    # Splitting datasets by class
    train_by_class, val_by_class, test_by_class = prepare_data(
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
    )

    # Building incremental steps
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

    # Run experiment
    history, model, optimizer, label_to_idx, idx_to_label = run_training(
        train_steps=train_steps,
        val_steps=val_steps,
        test_steps=test_steps,
        train_by_class=train_by_class,
        class_order=class_order,
        epochs=15,
    )

    plt.figure(figsize=(10, 5))

    # Plotta in ordine di class_order (così anche la legenda è ordinata)
    plotted = set()
    for idx, name in enumerate(class_order):
        if idx in history:
            accs = history[idx]
            y = np.asarray(accs, dtype=float)
            x = np.arange(1, len(y) + 1)
            plt.plot(x, y, label=name)
            plotted.add(idx)

    # Eventuali chiavi non previste (fallback)
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
