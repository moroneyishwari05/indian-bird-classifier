# 🦜 Indian Bird Sound Classifier

A deep learning pipeline that classifies **20 Indian bird species** from audio recordings using **EfficientNetB0** transfer learning and mel-spectrogram images.

## How It Works

1. **Downloads** bird call recordings from [Xeno-Canto](https://xeno-canto.org) (filtered to India, quality A)
2. **Splits** each recording into 5-second chunks (multiplies dataset size ~5–10×)
3. **Converts** audio chunks into 3-channel mel-spectrogram images (mel + delta + delta-delta)
4. **Trains** EfficientNetB0 (ImageNet pretrained) in two phases:
   - Phase 1: classification head only
   - Phase 2: full fine-tuning
5. **Evaluates** with classification report, confusion matrix, and training curves

## Species Covered

| Common Name | Scientific Name |
|---|---|
| Red-vented Bulbul | *Pycnonotus cafer* |
| Ashy Prinia | *Prinia socialis* |
| Spotted Dove | *Spilopelia senegalensis* |
| House Sparrow | *Passer domesticus* |
| House Crow | *Corvus splendens* |
| Common Myna | *Acridotheres tristis* |
| White-throated Kingfisher | *Halcyon smyrnensis* |
| Common Kingfisher | *Alcedo atthis* |
| Black Kite | *Milvus migrans* |
| Asian Koel | *Eudynamys scolopaceus* |
| Rose-ringed Parakeet | *Psittacula krameri* |
| Eurasian Hoopoe | *Upupa epops* |
| Black Drongo | *Dicrurus macrocercus* |
| Green Bee-eater | *Merops orientalis* |
| Indian Roller | *Coracias benghalensis* |
| Oriental Magpie-Robin | *Copsychus saularis* |
| Common Tailorbird | *Orthotomus sutoria* |
| Purple Sunbird | *Cinnyris asiaticus* |
| Jungle Babbler | *Turdoides striata* |
| Indian Peafowl | *Pavo cristatus* |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/indian-bird-classifier.git
cd indian-bird-classifier
```

### 2. Install dependencies
```bash
pip install requests librosa numpy pandas tqdm scikit-learn tensorflow matplotlib seaborn Pillow
```

### 3. Set your Xeno-Canto API key
Get your free API key at [xeno-canto.org](https://xeno-canto.org) → My Account → API key, then set it as an environment variable:

```bash
# Linux / macOS
export XENO_CANTO_API_KEY=your_key_here

# Windows (Command Prompt)
set XENO_CANTO_API_KEY=your_key_here
```

### 4. Run the pipeline
```bash
python efficientNet.py
```

This will download audio, extract spectrograms, train, and save the model — all automatically.

## Inference

```python
from efficientNet import predict_audio

results = predict_audio("path/to/bird_call.mp3", top_k=5)
for species, confidence in results:
    print(f"{species}: {confidence:.2%}")
```

## Output Files

```
bird_dataset/
├── raw_audio/          # Downloaded MP3s per species
├── chunks_npy/         # Mel-spectrogram .npy chunks
├── model/
│   ├── bird_efficientnet/   # Saved TF model
│   ├── label_encoder.pkl
│   └── config.json
└── results/
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── training_history.png
```

## Model Architecture

- **Backbone**: EfficientNetB0 (ImageNet pretrained, frozen in Phase 1)
- **Head**: GlobalAvgPool → Dropout(0.4) → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Softmax
- **Augmentation**: SpecAugment (frequency + time masking)
- **LR Schedule**: Cosine decay with linear warmup
- **Optimizer**: Adam

## Requirements

- Python 3.8+
- TensorFlow 2.x
- A Xeno-Canto account and API key (free)

## License

MIT
