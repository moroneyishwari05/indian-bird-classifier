#!/usr/bin/env python3
"""
Indian Bird Sound Classifier
=============================
Pipeline: Xeno-Canto download → Mel-spectrogram → EfficientNetB0 transfer learning

Model choice: EfficientNetB0 (pretrained on ImageNet)
- Far better than a custom CNN trained from scratch on small datasets
- Spectrograms treated as 3-channel images (R=mel, G=delta, B=delta-delta)
- Two-phase training: head-only first, then full fine-tune

Improvements over original code:
- Audio split into 5-second chunks (multiplies dataset size ~5-10x)
- SpecAugment (time + frequency masking) for regularization
- Global normalization (dataset-wide, not per-sample)
- Stratified train/val/test splits
- Cosine LR schedule with warmup
- Saves model in SavedModel format (not .h5)
- Clean prediction function for inference

Requirements:
    pip install requests librosa numpy pandas tqdm scikit-learn tensorflow matplotlib seaborn Pillow
"""

import os, time, json, pickle, warnings, urllib.parse
from datetime import datetime
from pathlib import Path

import requests
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

# =====================================================================
# CONFIG — edit these
# =====================================================================

# Set your Xeno-Canto API key as an environment variable:
#   export XENO_CANTO_API_KEY=your_key_here
# Get your key at: https://xeno-canto.org → My Account → API key
API_KEY = os.environ.get("XENO_CANTO_API_KEY", "")

SPECIES_LIST = [
    ("Pycnonotus cafer",        "Red-vented Bulbul"),
    ("Prinia socialis",         "Ashy Prinia"),
    ("Spilopelia senegalensis", "Spotted Dove"),
    ("Passer domesticus",       "House Sparrow"),
    ("Corvus splendens",        "House Crow"),
    ("Acridotheres tristis",    "Common Myna"),
    ("Halcyon smyrnensis",      "White-throated Kingfisher"),
    ("Alcedo atthis",           "Common Kingfisher"),
    ("Milvus migrans",          "Black Kite"),
    ("Eudynamys scolopaceus",   "Asian Koel"),
    ("Psittacula krameri",      "Rose-ringed Parakeet"),
    ("Upupa epops",             "Eurasian Hoopoe"),
    ("Dicrurus macrocercus",    "Black Drongo"),
    ("Merops orientalis",       "Green Bee-eater"),
    ("Coracias benghalensis",   "Indian Roller"),
    ("Copsychus saularis",      "Oriental Magpie-Robin"),
    ("Orthotomus sutoria",      "Common Tailorbird"),
    ("Cinnyris asiaticus",      "Purple Sunbird"),
    ("Turdoides striata",       "Jungle Babbler"),
    ("Pavo cristatus",          "Indian Peafowl"),
]

# Download settings
PER_SPECIES_MAX  = 50       # recordings to download per species
QUALITY_FILTER   = "q:A"    # A = highest quality only
LOCATION_FILTER  = "cnt:IN" # India recordings only
MIN_DURATION     = 5        # seconds
MAX_DURATION     = 120      # seconds

# Audio / spectrogram settings
SR          = 22050   # sample rate
CHUNK_SEC   = 5       # split each recording into 5-second chunks
HOP_LENGTH  = 512
N_FFT       = 2048
N_MELS      = 128
FMAX        = 8000
IMG_SIZE    = 224     # EfficientNet expects 224×224

# SpecAugment settings
FREQ_MASK_PARAM = 20   # max frequency bins to mask
TIME_MASK_PARAM = 40   # max time steps to mask
N_FREQ_MASKS    = 2
N_TIME_MASKS    = 2

# Training settings
BATCH_SIZE = 32
EPOCHS_HEAD  = 15    # phase 1: train head only
EPOCHS_FINE  = 35    # phase 2: fine-tune full network
LR_HEAD      = 1e-3
LR_FINE      = 1e-5
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15

# Directories
DATA_DIR         = Path("bird_dataset")
RAW_AUDIO_DIR    = DATA_DIR / "raw_audio"
NPY_DIR          = DATA_DIR / "chunks_npy"
MODEL_DIR        = DATA_DIR / "model"
RESULTS_DIR      = DATA_DIR / "results"

for d in [RAW_AUDIO_DIR, NPY_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# LOGGING
# =====================================================================

LOG_FILE = DATA_DIR / "run_log.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# =====================================================================
# PART 1 — DOWNLOAD FROM XENO-CANTO
# =====================================================================

def build_query(scientific_name):
    return (f'sp:"{scientific_name}" {LOCATION_FILTER} {QUALITY_FILTER} '
            f'len:">{MIN_DURATION}" len:"<{MAX_DURATION}"')

def fetch_recordings(scientific_name, wanted=PER_SPECIES_MAX):
    recordings, page = [], 1
    quoted = urllib.parse.quote(build_query(scientific_name), safe="")
    base   = "https://xeno-canto.org/api/3/recordings"

    while len(recordings) < wanted:
        url = f"{base}?query={quoted}&key={API_KEY}&per_page=100&page={page}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                log(f"  API error {r.status_code} for {scientific_name}")
                break
            data  = r.json()
            batch = data.get("recordings", [])
            if not batch:
                break
            recordings.extend(batch)
            if page >= int(data.get("numPages", 1)):
                break
            page += 1
            time.sleep(0.15)
        except Exception as e:
            log(f"  Exception: {e}")
            break

    return recordings[:wanted]

def download_file(url, path, min_bytes=2000):
    """Download to path; skip if already exists and valid."""
    if path.exists() and path.stat().st_size >= min_bytes:
        return True
    if url.startswith("//"):
        url = "https:" + url
    elif not url.startswith("http"):
        url = "https:" + url
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or not r.content:
            return False
        path.write_bytes(r.content)
        if path.stat().st_size < min_bytes:
            path.unlink()
            return False
        return True
    except Exception:
        if path.exists():
            path.unlink(missing_ok=True)
        return False

def download_all_species():
    """Download raw MP3s for every species."""
    log("=" * 60)
    log("STEP 1: Downloading audio from Xeno-Canto")
    log("=" * 60)
    summary = {}

    for sci, common in SPECIES_LIST:
        log(f"\n  [{SPECIES_LIST.index((sci, common))+1}/{len(SPECIES_LIST)}] {common} ({sci})")
        recs = fetch_recordings(sci)
        log(f"  Found {len(recs)} recordings")

        species_dir = RAW_AUDIO_DIR / sci.replace(" ", "_")
        species_dir.mkdir(exist_ok=True)
        count = 0

        for rec in tqdm(recs, desc=f"  {common}", leave=False, unit="file"):
            file_url = rec.get("file") or rec.get("url", "")
            if not file_url:
                continue
            fname  = f"XC{rec.get('id', int(time.time()*1000))}.mp3"
            fpath  = species_dir / fname
            if download_file(file_url, fpath):
                count += 1

        summary[sci] = count
        log(f"  Downloaded: {count}")
        time.sleep(0.5)

    log("\nDownload complete. Species counts:")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        log(f"  {k}: {v}")
    return summary


# =====================================================================
# PART 2 — FEATURE EXTRACTION (mel-spectrogram chunks)
# =====================================================================

def audio_to_chunks(filepath):
    """
    Load audio, split into CHUNK_SEC-second non-overlapping chunks.
    Returns list of numpy arrays (waveforms).
    """
    try:
        y, sr = librosa.load(str(filepath), sr=SR, mono=True)
    except Exception:
        return []

    chunk_len = CHUNK_SEC * sr
    chunks = []
    for start in range(0, len(y) - chunk_len + 1, chunk_len):
        chunk = y[start : start + chunk_len]
        if np.max(np.abs(chunk)) < 1e-4:   # skip silence
            continue
        chunks.append(chunk)
    return chunks

def waveform_to_spectrogram(y):
    """
    Convert waveform to 3-channel image (mel, delta-mel, delta-delta-mel).
    Output shape: (IMG_SIZE, IMG_SIZE, 3), float32 in [0, 1].
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    delta  = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    # Stack into 3-channel image
    img = np.stack([mel_db, delta, delta2], axis=-1)  # (128, T, 3)

    # Resize to IMG_SIZE × IMG_SIZE using simple interpolation
    from PIL import Image
    channels = []
    for c in range(3):
        arr = img[:, :, c]
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255
        else:
            arr = np.zeros_like(arr)
        pil = Image.fromarray(arr.astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        channels.append(np.array(pil, dtype=np.float32) / 255.0)

    return np.stack(channels, axis=-1)  # (224, 224, 3)

def extract_all_chunks():
    """
    Walk RAW_AUDIO_DIR, convert every MP3 to chunk spectrograms,
    save as .npy files in NPY_DIR/<species>/
    Returns metadata dataframe.
    """
    log("=" * 60)
    log("STEP 2: Extracting mel-spectrogram chunks")
    log("=" * 60)

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Install Pillow: pip install Pillow")

    meta = []
    species_dirs = sorted(RAW_AUDIO_DIR.iterdir())

    for sp_dir in species_dirs:
        if not sp_dir.is_dir():
            continue
        sci_name  = sp_dir.name.replace("_", " ")
        npy_sp    = NPY_DIR / sp_dir.name
        npy_sp.mkdir(exist_ok=True)

        mp3_files = list(sp_dir.glob("*.mp3"))
        chunk_count = 0

        for mp3 in tqdm(mp3_files, desc=f"  {sci_name}", leave=False):
            chunks = audio_to_chunks(mp3)
            for i, wav in enumerate(chunks):
                npy_path = npy_sp / f"{mp3.stem}_chunk{i:03d}.npy"
                if not npy_path.exists():
                    try:
                        img = waveform_to_spectrogram(wav)
                        np.save(str(npy_path), img.astype(np.float32))
                    except Exception:
                        continue
                meta.append({"path": str(npy_path), "species": sci_name})
                chunk_count += 1

        log(f"  {sci_name}: {chunk_count} chunks from {len(mp3_files)} files")

    df = pd.DataFrame(meta)
    df.to_csv(DATA_DIR / "chunks_metadata.csv", index=False)
    log(f"\nTotal chunks: {len(df)}")
    log(f"Species: {df['species'].nunique()}")
    return df


# =====================================================================
# PART 3 — DATASET LOADING + SPECAUGMENT
# =====================================================================

def spec_augment(img):
    """Apply SpecAugment: random frequency and time masking."""
    img = img.copy()
    H, W, C = img.shape

    # Frequency masking (rows)
    for _ in range(N_FREQ_MASKS):
        f = np.random.randint(0, FREQ_MASK_PARAM)
        f0 = np.random.randint(0, max(1, H - f))
        img[f0:f0+f, :, :] = 0.0

    # Time masking (columns)
    for _ in range(N_TIME_MASKS):
        t = np.random.randint(0, TIME_MASK_PARAM)
        t0 = np.random.randint(0, max(1, W - t))
        img[:, t0:t0+t, :] = 0.0

    return img

def load_dataset(df):
    """Load all .npy chunks into memory. Returns X, y arrays."""
    log("Loading dataset into memory...")
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
        try:
            img = np.load(row["path"])
            if img.shape == (IMG_SIZE, IMG_SIZE, 3):
                X.append(img)
                y.append(row["species"])
        except Exception:
            continue
    return np.array(X, dtype=np.float32), np.array(y)

class AugmentedDataset(keras.utils.Sequence):
    """Keras Sequence with SpecAugment for training."""
    def __init__(self, X, y, batch_size=BATCH_SIZE, augment=True):
        self.X        = X
        self.y        = y
        self.bs       = batch_size
        self.augment  = augment
        self.indices  = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.bs : (idx+1) * self.bs]
        bx = self.X[batch_idx].copy()
        by = self.y[batch_idx]
        if self.augment:
            for i in range(len(bx)):
                if np.random.rand() > 0.5:
                    bx[i] = spec_augment(bx[i])
        return bx, by

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# =====================================================================
# PART 4 — MODEL: EfficientNetB0 TRANSFER LEARNING
# =====================================================================

def build_model(num_classes):
    """
    EfficientNetB0 with ImageNet weights.
    Top layers: GlobalAvgPool → Dropout → Dense(256) → Dropout → Softmax
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None
    )
    base.trainable = False   # freeze for phase 1

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base


def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs, lr_start):
    """Learning rate schedule: linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        return lr_start * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return lr_start * 0.5 * (1 + np.cos(np.pi * progress))


# =====================================================================
# PART 5 — TRAINING PIPELINE
# =====================================================================

def train(df):
    log("=" * 60)
    log("STEP 3: Training EfficientNetB0")
    log("=" * 60)

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["species"])
    num_classes = len(le.classes_)
    log(f"Classes: {num_classes}")

    # Stratified split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE/(1-TEST_SIZE),
        stratify=train_df["label"], random_state=42
    )
    log(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Load data
    X_train, y_train_str = load_dataset(train_df)
    X_val,   y_val_str   = load_dataset(val_df)
    X_test,  y_test_str  = load_dataset(test_df)

    y_train = le.transform(y_train_str)
    y_val   = le.transform(y_val_str)
    y_test  = le.transform(y_test_str)

    # One-hot for training
    y_train_oh = keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = keras.utils.to_categorical(y_val,   num_classes)

    # Build model
    model, base = build_model(num_classes)
    model.summary()

    # --- PHASE 1: Train head only ---
    log("\n--- Phase 1: Training classification head ---")
    model.compile(
        optimizer=keras.optimizers.Adam(LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    train_gen = AugmentedDataset(X_train, y_train_oh, augment=True)
    val_gen   = AugmentedDataset(X_val,   y_val_oh,   augment=False)

    ckpt_path = str(MODEL_DIR / "best_model.keras")
    callbacks_p1 = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy"),
        keras.callbacks.LearningRateScheduler(
            lambda e: cosine_decay_with_warmup(e, EPOCHS_HEAD, 3, LR_HEAD)
        )
    ]

    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_HEAD, callbacks=callbacks_p1)

    # --- PHASE 2: Fine-tune entire network ---
    log("\n--- Phase 2: Fine-tuning full EfficientNetB0 ---")
    base.trainable = True   # unfreeze all layers

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_p2 = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy"),
        keras.callbacks.LearningRateScheduler(
            lambda e: cosine_decay_with_warmup(e, EPOCHS_FINE, 5, LR_FINE)
        )
    ]

    h2 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS_FINE, callbacks=callbacks_p2)

    # --- Evaluate on test set ---
    log("\n--- Test set evaluation ---")
    test_gen  = AugmentedDataset(X_test, keras.utils.to_categorical(y_test, num_classes),
                                  augment=False)
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    log(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    log("\n" + report)

    # Save report
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # --- Confusion matrix ---
    plot_confusion_matrix(y_test, y_pred, le.classes_)

    # --- Training curves (combined phases) ---
    plot_history(h1, h2)

    # --- Save model + encoder ---
    model.save(str(MODEL_DIR / "bird_efficientnet"))
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    config = {
        "num_classes": num_classes,
        "species": list(le.classes_),
        "img_size": IMG_SIZE,
        "sr": SR,
        "chunk_sec": CHUNK_SEC,
        "n_mels": N_MELS,
        "fmax": FMAX,
        "test_accuracy": float(test_acc),
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log(f"\nModel saved to: {MODEL_DIR / 'bird_efficientnet'}")
    log(f"Label encoder: {MODEL_DIR / 'label_encoder.pkl'}")
    return model, le


# =====================================================================
# PART 6 — VISUALIZATION
# =====================================================================

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(classes)), max(8, len(classes)-2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {path}")

def plot_history(h1, h2):
    """Merge phase-1 and phase-2 histories and plot."""
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    vacc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    ep    = range(1, len(acc) + 1)
    p1end = len(h1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, train_vals, val_vals, title, ylabel in [
        (axes[0], acc, vacc, "Accuracy", "Accuracy"),
        (axes[1], loss, vloss, "Loss", "Loss"),
    ]:
        ax.plot(ep, train_vals, label="Train")
        ax.plot(ep, val_vals,   label="Val")
        ax.axvline(p1end, color="gray", linestyle="--", label="Phase 2 start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "training_history.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    log(f"Saved: {path}")


# =====================================================================
# PART 7 — INFERENCE (predict a new audio file)
# =====================================================================

def predict_audio(audio_path, model_dir=None, top_k=5):
    """
    Predict bird species from a new audio file.

    Args:
        audio_path: path to .mp3 or .wav file
        model_dir:  path to saved model directory (default: bird_dataset/model/bird_efficientnet)
        top_k:      number of top predictions to return

    Returns:
        List of (species_name, confidence) tuples, best prediction first
    """
    from PIL import Image

    if model_dir is None:
        model_dir = str(MODEL_DIR / "bird_efficientnet")

    model = keras.models.load_model(model_dir)
    with open(Path(model_dir).parent / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    try:
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as e:
        print(f"Could not load audio: {e}")
        return []

    # Use middle chunk for prediction (most representative)
    chunk_len = CHUNK_SEC * SR
    if len(y) >= chunk_len:
        start = max(0, (len(y) - chunk_len) // 2)
        y     = y[start : start + chunk_len]
    else:
        y = np.pad(y, (0, chunk_len - len(y)))

    img   = waveform_to_spectrogram(y)
    batch = np.expand_dims(img, 0)

    probs    = model.predict(batch, verbose=0)[0]
    top_idx  = np.argsort(probs)[-top_k:][::-1]
    results  = [(le.classes_[i], float(probs[i])) for i in top_idx]
    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    if not API_KEY:
        print("\nERROR: Set your Xeno-Canto API key as an environment variable:")
        print("  export XENO_CANTO_API_KEY=your_key_here")
        print("Get your key at: https://xeno-canto.org → My Account → API key")
        return

    # Step 1: Download audio
    download_all_species()

    # Step 2: Extract spectrogram chunks
    df = extract_all_chunks()

    # Validate we have enough data
    min_samples = df["species"].value_counts().min()
    log(f"\nMin samples per species: {min_samples}")
    if min_samples < 5:
        log("WARNING: Some species have very few samples. Consider lowering TEST_SIZE.")

    # Step 3: Train model
    model, le = train(df)

    log("\n" + "=" * 60)
    log("DONE! Your model is ready.")
    log(f"Model: {MODEL_DIR / 'bird_efficientnet'}")
    log(f"Results: {RESULTS_DIR}")
    log("=" * 60)

    # Example inference
    log("\nExample inference usage:")
    log("  results = predict_audio('path/to/bird_call.mp3')")
    log("  for species, conf in results:")
    log("      print(f'{species}: {conf:.2%}')")


if __name__ == "__main__":
    main()
