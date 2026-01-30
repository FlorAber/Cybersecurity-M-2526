import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pickle
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time

# Configurazione
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

SOURCE_PATH = "../data/processed_data_fix/"  # Percorso ai dati preprocessati NON PRESENTI NELLA REPO PERCHE TROPPO GRANDI
PRETRAINED_MODEL_PATH = (
    "./checkpoints/checkpoint_us.pt"  # Percorso al modello pre-addestrato
)

POISON_RATES_TO_TEST = [0.001, 0.005, 0.01, 0.02, 0.05]
NUM_STEPS = 5
# ============================================================================
# PARAMETRI BACKDOOR
# ============================================================================
TARGET_CLASSES = ["DDoS", "Injection"]
BACKDOOR_CLASS = "Normal"


# ============================================================================
# CLASSI
# ============================================================================
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


class IntelligentBackdoorTrigger:
    def __init__(
        self,
        feature_indices: List[int] = None,
        trigger_values: List[float] = None,
        trigger_type: str = "adaptive",
    ):
        self.feature_indices = feature_indices or [5, 10, 15]
        self.trigger_values = trigger_values or [3.5, 3.5, 3.5]
        self.trigger_type = trigger_type

    def apply(self, X: np.ndarray) -> np.ndarray:
        X_poisoned = X.copy()

        if self.trigger_type == "adaptive":
            for i, idx in enumerate(self.feature_indices):
                if idx < X.shape[1]:
                    trigger_val = np.percentile(X[:, idx], 95) + self.trigger_values[i]
                    X_poisoned[:, idx] = trigger_val
        elif self.trigger_type == "stochastic":
            for i, idx in enumerate(self.feature_indices):
                if idx < X.shape[1]:
                    noise = np.random.normal(0, 0.1 * self.trigger_values[i], len(X))
                    X_poisoned[:, idx] = self.trigger_values[i] + noise
        elif self.trigger_type == "pattern":
            for i, idx in enumerate(self.feature_indices):
                if idx < X.shape[1]:
                    X_poisoned[:, idx] = self.trigger_values[i]

        return X_poisoned


class StrategicDataPoisoner:
    def __init__(
        self,
        trigger: IntelligentBackdoorTrigger,
        target_classes: List[str] = None,
        backdoor_class: str = "DDoS",
        poison_rate: float = 0.15,
        strategy: str = "easy",
    ):
        self.trigger = trigger
        self.target_classes = target_classes or ["Normal"]
        self.backdoor_class = backdoor_class
        self.poison_rate = poison_rate
        self.strategy = strategy
        self.poisoned_samples = 0

    def select_samples_for_poisoning(
        self, X: np.ndarray, y: np.ndarray, model=None, device="cpu"
    ) -> np.ndarray:
        n_poison = max(1, int(len(X) * self.poison_rate))

        if self.strategy == "random":
            return np.random.choice(len(X), n_poison, replace=False)

        elif self.strategy == "easy":
            if model is not None:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    entropy = entropy.cpu().numpy()

                sorted_idx = np.argsort(entropy)[::-1]
                return sorted_idx[:n_poison]
            else:
                return np.random.choice(len(X), n_poison, replace=False)

        else:
            return np.random.choice(len(X), n_poison, replace=False)

    def poison_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_encoder,
        backdoor_idx: int,
        model=None,
        device="cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:

        y_str = label_encoder.inverse_transform(y)

        X_result = X.copy()
        y_result = y.copy()

        # Avvelena TUTTE le target_classes
        for target_class in self.target_classes:
            target_mask = y_str == target_class
            X_target = X[target_mask]
            y_target = y[target_mask]
            target_indices_full = np.where(target_mask)[0]

            if len(X_target) == 0:
                continue

            selected_indices = self.select_samples_for_poisoning(
                X_target, y_target, model, device
            )

            X_poisoned = self.trigger.apply(X_target[selected_indices])
            y_poisoned = np.full(len(selected_indices), backdoor_idx)

            poisoned_full_indices = target_indices_full[selected_indices]
            X_result[poisoned_full_indices] = X_poisoned
            y_result[poisoned_full_indices] = y_poisoned

            self.poisoned_samples += len(selected_indices)

        return X_result, y_result


# ============================================================================
# FUNZIONI SUPPORTO
# ============================================================================
def load_datasets():
    print("Loading datasets...")
    X_train = np.load(SOURCE_PATH + "X_train.npy")
    y_train = np.load(SOURCE_PATH + "y_train.npy")
    X_val = np.load(SOURCE_PATH + "X_val.npy")
    y_val = np.load(SOURCE_PATH + "y_val.npy")
    X_test = np.load(SOURCE_PATH + "X_test.npy")
    y_test = np.load(SOURCE_PATH + "y_test.npy")

    with open(SOURCE_PATH + "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(SOURCE_PATH + "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(SOURCE_PATH + "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        label_encoder,
        feature_names,
    )


def load_pretrained_model(model_path: str, device: str = "cpu"):
    print(f"Loading pretrained model from {model_path}...")

    try:
        import numpy as np

        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except:
        pass

    try:
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
    except:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

    input_dim = checkpoint["extra"]["input_dim"]
    num_classes = checkpoint["extra"]["num_classes"]
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = checkpoint["idx_to_label"]

    model = IDSModel(input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    return model, label_to_idx, idx_to_label, input_dim, num_classes


def train_epoch(model, X, y, optimizer, criterion, device="cpu", batch_size=2048):
    if len(X) == 0:
        return 0.0

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, X, y, device="cpu"):
    if len(X) == 0:
        return np.array([]), 0.0

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    return preds, accuracy_score(y, preds)


# ============================================================================
# TEST MULTIPLI POISON RATES
# ============================================================================


def test_multiple_poison_rates():
    """
    Versione CORRETTA con mapping giusto
    """
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE POISON RATES - CORRECTED VERSION")
    print("=" * 80)
    print(f"Target classes: {TARGET_CLASSES}")
    print(f"Backdoor class: {BACKDOOR_CLASS}")
    print(f"Poison rates: {[f'{r:.1%}' for r in POISON_RATES_TO_TEST]}")
    print(f"Steps per rate: {NUM_STEPS}")
    print("=" * 80 + "\n")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        label_encoder,
        feature_names,
    ) = load_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    y_train_str = label_encoder.inverse_transform(y_train)
    y_test_str = label_encoder.inverse_transform(y_test)

    train_by_class = {}
    test_by_class = {}
    for cls in np.unique(y_train_str):
        train_by_class[cls] = X_train[y_train_str == cls]
    for cls in np.unique(y_test_str):
        test_by_class[cls] = X_test[y_test_str == cls]

    backdoor_idx = label_encoder.transform([BACKDOOR_CLASS])[0]
    print(f"Backdoor class '{BACKDOOR_CLASS}' has index: {backdoor_idx}\n")

    all_results = {}

    for poison_rate in POISON_RATES_TO_TEST:
        print(f"\n{'='*80}")
        print(f"TESTING POISON RATE: {poison_rate:.1%}")
        print(f"{'='*80}")

        start_time = time.time()

        model, label_to_idx, idx_to_label, input_dim, num_classes = (
            load_pretrained_model(PRETRAINED_MODEL_PATH, device)
        )

        trigger_features = [5, 10, 15]
        trigger = IntelligentBackdoorTrigger(
            feature_indices=trigger_features,
            trigger_values=[3.5, 3.5, 3.5],
            trigger_type="adaptive",
        )

        poisoner = StrategicDataPoisoner(
            trigger=trigger,
            target_classes=TARGET_CLASSES,
            backdoor_class=BACKDOOR_CLASS,
            poison_rate=poison_rate,
            strategy="easy",
        )

        history = defaultdict(list)
        backdoor_history = []

        # Baseline
        print(f"\nBaseline evaluation:")
        for cls in idx_to_label:
            if cls in test_by_class:
                X_cls = test_by_class[cls]
                y_cls_idx = label_encoder.transform([cls])[0]
                y_cls_full = np.full(len(X_cls), y_cls_idx)
                preds, acc = evaluate_model(model, X_cls, y_cls_full, device)
                history[cls].append(acc)

        # Baseline ASR per ogni target class
        asr_baseline = {}
        for target_cls in TARGET_CLASSES:
            if target_cls in test_by_class:
                X_test_target = test_by_class[target_cls]
                X_test_bd = trigger.apply(X_test_target)
                y_test_bd = np.full(len(X_test_bd), backdoor_idx)
                preds, _ = evaluate_model(model, X_test_bd, y_test_bd, device)
                asr = np.mean(preds == backdoor_idx)
                asr_baseline[target_cls] = asr
                print(f"  Baseline ASR ({target_cls} -> {BACKDOOR_CLASS}): {asr:.4f}")

        backdoor_history.append(asr_baseline)

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        # Poisoning
        print(f"\nPoisoning (poison_rate={poison_rate:.1%}):")
        for step in range(1, NUM_STEPS + 1):
            X_tr_list, y_tr_list = [], []
            for cls in idx_to_label:
                if cls in train_by_class:
                    X_cls = train_by_class[cls]
                    y_cls = np.full(len(X_cls), label_encoder.transform([cls])[0])
                    X_tr_list.append(X_cls)
                    y_tr_list.append(y_cls)

            X_tr = np.vstack(X_tr_list) if X_tr_list else np.array([])
            y_tr = np.hstack(y_tr_list) if y_tr_list else np.array([])

            if len(X_tr) == 0:
                continue

            X_tr_poisoned, y_tr_poisoned = poisoner.poison_dataset(
                X_tr, y_tr, label_encoder, backdoor_idx, model, device
            )

            for epoch in range(2):
                train_epoch(
                    model,
                    X_tr_poisoned,
                    y_tr_poisoned,
                    optimizer,
                    criterion,
                    device,
                    batch_size=256,
                )

            for cls in idx_to_label:
                if cls in test_by_class:
                    X_cls = test_by_class[cls]
                    y_cls_idx = label_encoder.transform([cls])[0]
                    y_cls_full = np.full(len(X_cls), y_cls_idx)
                    preds, acc = evaluate_model(model, X_cls, y_cls_full, device)
                    history[cls].append(acc)

            # ASR
            asr_current = {}
            for target_cls in TARGET_CLASSES:
                if target_cls in test_by_class:
                    X_test_target = test_by_class[target_cls]
                    X_test_bd = trigger.apply(X_test_target)
                    y_test_bd = np.full(len(X_test_bd), backdoor_idx)
                    preds, _ = evaluate_model(model, X_test_bd, y_test_bd, device)
                    asr = np.mean(preds == backdoor_idx)
                    asr_current[target_cls] = asr

            backdoor_history.append(asr_current)

            if step % 2 == 0 or step == 1:
                asr_str = " | ".join(
                    [f"{cls}={asr_current[cls]:.4f}" for cls in TARGET_CLASSES]
                )
                print(f"  Step {step:2d}: ASR {asr_str}")

        elapsed = time.time() - start_time

        final_asr_dict = backdoor_history[-1]
        final_asr_mean = np.mean(list(final_asr_dict.values()))
        baseline_asr_dict = backdoor_history[0]
        baseline_asr_mean = np.mean(list(baseline_asr_dict.values()))

        all_results[poison_rate] = {
            "history": dict(history),
            "backdoor_history": backdoor_history,
            "poisoned_samples": poisoner.poisoned_samples,
            "final_asr": final_asr_mean,
            "final_asr_dict": final_asr_dict,
            "baseline_asr": baseline_asr_mean,
            "baseline_asr_dict": baseline_asr_dict,
            "asr_improvement": final_asr_mean - baseline_asr_mean,
            "time": elapsed,
        }

        print(f"\nResults for poison_rate={poison_rate:.1%}:")
        print(f"  Final ASR (mean): {final_asr_mean:.4f}")
        for cls in TARGET_CLASSES:
            print(f"    {cls} -> {BACKDOOR_CLASS}: {final_asr_dict[cls]:.4f}")
        print(f"  ASR Improvement: {final_asr_mean - baseline_asr_mean:+.4f}")
        print(f"  Poisoned samples: {poisoner.poisoned_samples}")
        print(f"  Time: {elapsed:.1f}s")

    return all_results


# ============================================================================
# PLOT
# ============================================================================


def plot_results(all_results: Dict):
    """
    3 Grafici come richiesto:
    1. ASR Evolution per Class per Different Poison Rates
    2. Accuracy per Class per Different Poison Rates
    3. Summary Table
    """
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    fig = plt.figure(figsize=(20, 14))

    poison_rates = sorted(all_results.keys())

    ax = plt.subplot(2, 2, 1)

    poison_colors = plt.cm.tab10(range(len(poison_rates)))
    poison_color_map = dict(zip(poison_rates, poison_colors))
    colors_class = {
        "DDoS": "#e74c3c",
        "Injection": "#3498db",
        "Normal": "#2ecc71",
        "Probe": "#f39c12",
        "Scanning": "#9b59b6",
    }
    linestyles = ["-", "--", ":", "-."]
    class_linestyle = {
        cls: linestyles[i % len(linestyles)] for i, cls in enumerate(TARGET_CLASSES)
    }
    color_factors = np.linspace(0.5, 1.0, len(poison_rates))

    for pr_idx, pr in enumerate(poison_rates):
        backdoor_history = all_results[pr]["backdoor_history"]

        for target_cls in TARGET_CLASSES:
            asr_per_step = [d[target_cls] for d in backdoor_history]

            ax.plot(
                range(len(asr_per_step)),
                asr_per_step,
                marker="o",
                label=f"{target_cls} ({pr:.1%})",
                color=poison_color_map[pr],
                linewidth=2,
                linestyle=class_linestyle[target_cls],
                alpha=0.85,
            )

    ax.set_xlabel("Poisoning Step", fontsize=12, fontweight="bold")
    ax.set_ylabel("ASR", fontsize=12, fontweight="bold")
    ax.set_title(
        "ASR Evolution per Class for Different Poison Rates",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = plt.subplot(2, 2, 2)

    class_list = list(all_results[poison_rates[0]]["history"].keys())
    x = np.arange(len(poison_rates))
    width = 0.10

    colors_class_bar = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, cls in enumerate(class_list):
        accs = [all_results[pr]["history"][cls][-1] for pr in poison_rates]
        offset = width * (i - len(class_list) / 2 + 0.5)
        ax.bar(
            x + offset,
            accs,
            width,
            label=cls,
            color=colors_class_bar[i % len(colors_class_bar)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Poison Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Final Accuracy per Class for Different Poison Rates",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pr:.1%}" for pr in poison_rates], fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    ax = plt.subplot(2, 2, (3, 4))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    table_data.append(
        [
            "Poison Rate",
            "ASR Baseline",
            "ASR Final",
            "ASR Improvement",
            "Poisoned Samples",
        ]
    )

    for pr in poison_rates:
        baseline = all_results[pr]["baseline_asr"]
        final = all_results[pr]["final_asr"]
        improvement = all_results[pr]["asr_improvement"]
        samples = all_results[pr]["poisoned_samples"]

        table_data.append(
            [
                f"{pr:.1%}",
                f"{baseline:.4f}",
                f"{final:.4f}",
                f"{improvement:+.4f}",
                f"{samples:,}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.2, 0.2, 0.2, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(weight="bold", color="white", fontsize=12)

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):

            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ecf0f1")
            else:
                table[(i, j)].set_facecolor("white")

            if j == 3:
                table[(i, j)].set_facecolor("#fff3cd")

    ax.set_title(
        "Summary Table: ASR and Poisoning Statistics",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig("poison_rates_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: poison_rates_comparison.png")
    plt.show()


def print_summary(all_results: Dict):
    """
    Stampa summary
    """
    print("\n" + "=" * 80)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 80)

    poison_rates = sorted(all_results.keys())

    print(
        f"\n{'Poison Rate':<15} {'Baseline ASR':<15} {'Final ASR':<15} {'Improvement':<15} {'Samples':<12}"
    )
    print("-" * 75)

    for pr in poison_rates:
        results = all_results[pr]
        print(
            f"{pr:>6.1%}{'':<8} {results['baseline_asr']:>6.4f}{'':<8} {results['final_asr']:>6.4f}{'':<8} {results['asr_improvement']:>+6.4f}{'':<8} {results['poisoned_samples']:>8}"
        )

    print(f"\n{'='*80}")
    print("ASR PER TARGET CLASS")
    print(f"{'='*80}")

    for pr in poison_rates:
        print(f"\nPoison Rate: {pr:.1%}")
        final_asr_dict = all_results[pr]["final_asr_dict"]
        for target_cls in TARGET_CLASSES:
            asr = final_asr_dict[target_cls]
            print(f"  {target_cls} -> {BACKDOOR_CLASS}: {asr:.4f}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "=" * 80)
    print("MULTIPLE POISON RATES TESTING - CORRECTED VERSION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Target Classes: {TARGET_CLASSES}")
    print(f"  Backdoor Class: {BACKDOOR_CLASS}")
    print("=" * 80)

    all_results = test_multiple_poison_rates()
    plot_results(all_results)
    print_summary(all_results)

    import json

    results_to_save = {}
    for pr, res in all_results.items():
        results_to_save[f"{pr:.1%}"] = {
            "baseline_asr": float(res["baseline_asr"]),
            "final_asr": float(res["final_asr"]),
            "asr_improvement": float(res["asr_improvement"]),
            "poisoned_samples": int(res["poisoned_samples"]),
            "time": float(res["time"]),
            "asr_per_class": {
                str(k): float(v) for k, v in res["final_asr_dict"].items()
            },
        }

    with open("poison_rates_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Results saved to poison_rates_results.json")

    print("\n" + "=" * 80)
    print("TESTING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
