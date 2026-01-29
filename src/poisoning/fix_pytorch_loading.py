"""
FIX PER PYTORCH 2.6+ - CARICAMENTO MODELLI
===========================================

Questo script risolve l'errore:
"Weights only load failed. This file can still be loaded..."
"""

import torch
import pickle
import numpy as np

# ============================================================================
# OPZIONE 1: Usa questo per caricare il modello (CONSIGLIATO)
# ============================================================================

def load_model_safe(model_path: str, device: str = "cpu"):
    """
    Carica un modello in modo sicuro con PyTorch 2.6+
    """
    print(f"Loading model from {model_path}...")
    print(f"Device: {device}")
    
    # Step 1: Aggiungi numpy ai safe globals
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except:
        try:
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        except:
            print("Warning: Couldn't add numpy to safe globals (non-critical)")
    
    # Step 2: Prova con pickle first (più sicuro)
    try:
        print("Trying pickle...")
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print("✓ Successfully loaded with pickle")
        return checkpoint
    except Exception as e:
        print(f"  Pickle failed: {e}")
    
    # Step 3: Prova con torch.load e weights_only=False
    try:
        print("Trying torch.load with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("✓ Successfully loaded with torch.load (weights_only=False)")
        return checkpoint
    except TypeError:
        # Versioni vecchie di PyTorch non supportano weights_only
        print("Trying torch.load without weights_only parameter...")
        checkpoint = torch.load(model_path, map_location=device)
        print("✓ Successfully loaded with torch.load")
        return checkpoint
    except Exception as e:
        print(f"✗ ERRORE: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}")


# ============================================================================
# OPZIONE 2: Usa questa versione con context manager (MÀ ROBUSTO)
# ============================================================================

def load_model_with_context(model_path: str, device: str = "cpu"):
    """
    Usa context manager per numpy safe globals
    """
    print(f"Loading model from {model_path}...")
    
    # Prova con pickle
    try:
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print("✓ Loaded with pickle")
        return checkpoint
    except:
        pass
    
    # Prova con context manager
    try:
        import numpy as np
        with torch.serialization.safe_globals([np._core.multiarray.scalar]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("✓ Loaded with torch.load + safe_globals")
        return checkpoint
    except TypeError:
        # Fallback per versioni vecchie
        checkpoint = torch.load(model_path, map_location=device)
        print("✓ Loaded with torch.load (old version)")
        return checkpoint
    except Exception as e:
        print(f"✗ ERRORE: {e}")
        raise


# ============================================================================
# OPZIONE 3: Check la versione di PyTorch e agisci di conseguenza
# ============================================================================

def get_pytorch_version():
    """Ritorna la versione di PyTorch"""
    import re
    version_str = torch.__version__
    match = re.search(r'(\d+)\.(\d+)', version_str)
    if match:
        major, minor = int(match.group(1)), int(match.group(2))
        return (major, minor)
    return None


def load_model_smart(model_path: str, device: str = "cpu"):
    """
    Carica il modello in modo intelligente basato su versione PyTorch
    """
    version = get_pytorch_version()
    print(f"PyTorch version: {torch.__version__} → {version}")
    
    # PyTorch 2.6+
    if version and version >= (2, 6):
        print("Detected PyTorch 2.6+, using weights_only=False...")
        try:
            # Aggiungi numpy ai safe globals
            import numpy as np
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        except:
            pass
        
        return torch.load(model_path, map_location=device, weights_only=False)
    
    # PyTorch 2.0-2.5
    elif version and version >= (2, 0):
        print("Detected PyTorch 2.0-2.5, trying standard load...")
        return torch.load(model_path, map_location=device)
    
    # PyTorch < 2.0
    else:
        print("Detected old PyTorch, using standard load...")
        return torch.load(model_path, map_location=device)


# ============================================================================
# INTEGRAZIONE NEL TUO CODICE
# ============================================================================

# Nel file improved_backdoor_attack.py, sostituisci:
#
#   checkpoint = torch.load(model_path, map_location=device)
#
# Con:
#
#   checkpoint = load_model_safe(model_path, device)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PYTORCH 2.6+ MODEL LOADING FIX - TEST")
    print("="*70)
    
    print(f"\nTesting load functions...")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test path
    model_path = "../src/ids_checkpoint.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nTrying to load: {model_path}")
    print(f"Device: {device}")
    
    try:
        checkpoint = load_model_safe(model_path, device)
        print("\n✅ SUCCESS!")
        print(f"Checkpoint keys: {checkpoint.keys()}")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        print("\nSoluzioni:")
        print("1. Assicurati che il path sia corretto")
        print("2. Assicurati che il file non sia corrotto")
        print("3. Prova a rigenerare il modello")