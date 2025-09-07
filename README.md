# Deep Learning Sanskrit Text-to-Speech Synthesizer

An advanced end-to-end Text-to-Speech (TTS) system for Sanskrit using deep learning techniques. The system converts Unicode Sanskrit text into natural-sounding speech with high intelligibility.

## Features

- **Text Front-End Processing**
  - Sanskrit text normalization (numbers, dates, symbols, abbreviations)
  - Grapheme-to-Phoneme (G2P) conversion following Sanskrit phonetic rules
  - Support for Devanagari script and Unicode normalization

- **Deep Learning Models**
  - Tacotron 2 acoustic model for mel-spectrogram generation
  - HiFi-GAN vocoder for high-fidelity audio synthesis
  - PyTorch implementation with GPU acceleration

- **End-to-End Pipeline**
  - Seamless integration of all components
  - Batch processing capabilities
  - Interactive CLI interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sanskrit-tts
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models (if available):
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download models (replace with actual URLs)
wget <tacotron2-model-url> -O checkpoints/tacotron2_sanskrit.pth
wget <hifigan-model-url> -O checkpoints/hifigan_sanskrit.pth
```

## Usage

### Command Line Interface

#### Single Text Synthesis
```bash
python src/main.py --text "नमस्ते" --output hello.wav
```

#### Batch Processing
```bash
python src/main.py --input_file texts.txt --output_dir ./outputs/
```

#### Interactive Mode
```bash
python src/main.py --interactive
```

### Python API

```python
from src.inference.tts_pipeline import SanskritTTSPipeline

# Initialize pipeline
pipeline = SanskritTTSPipeline(
    tacotron2_checkpoint="checkpoints/tacotron2_sanskrit.pth",
    hifigan_checkpoint="checkpoints/hifigan_sanskrit.pth"
)

# Synthesize speech
audio, sample_rate = pipeline.synthesize("संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति।")

# Save to file
import soundfile as sf
sf.write("output.wav", audio, sample_rate)
```

## Training

### Data Preparation

1. Prepare your Sanskrit speech dataset with the following structure:

anskrit-tts/
├── src/
│   ├── text_frontend/
│   │   ├── text_normalizer.py      # Text normalization
│   │   └── g2p_converter.py        # G2P conversion
│   ├── models/
│   │   ├── tacotron2.py           # Tacotron 2 model
│   │   └── hifigan.py             # HiFi-GAN model
│   ├── training/
│   │   ├── train_tacotron2.py     # Tacotron 2 training
│   │   └── train_hifigan.py       # HiFi-GAN training
│   ├── inference/
│   │   └── tts_pipeline.py        # Inference pipeline
│   └── main.py                    # Main CLI application
├── config/
│   └── model_config.yaml          # Model configuration
├── requirements.txt               # Dependencies
└── README.md                     # This file

2. Format of filelist files:


### Training Tacotron 2

```bash
python src/training/train_tacotron2.py \
    --output_directory ./tacotron2_output \
    --log_directory ./logs
```

### Training HiFi-GAN

```bash
python src/training/train_hifigan.py \
    --config config/model_config.yaml \
    --input_mels_dir ./tacotron2_output/mels \
    --input_wavs_dir ./data/wavs
```

## Project Structure



## Technical Details

### Text Processing

- **Normalization**: Expands numbers, dates, symbols, and abbreviations into full word forms
- **G2P Conversion**: Maps Sanskrit graphemes to phonemes following traditional phonetic rules
- **Sandhi Rules**: Applies basic Sanskrit phonetic combination rules

### Models

- **Tacotron 2**: Sequence-to-sequence model generating mel-spectrograms from phoneme sequences
- **HiFi-GAN**: Neural vocoder synthesizing high-fidelity audio from mel-spectrograms

### Audio Specifications

- Sample Rate: 22,050 Hz
- Mel Channels: 80
- FFT Size: 1024
- Hop Size: 256
- Window Size: 1024

## Requirements

- Python 3.7+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for complete list

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Tacotron 2: [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
- HiFi-GAN: [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
- Sanskrit linguistic resources and phonetic rules

## 🔧 Quick Start

```bash
# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Single text synthesis
python src/main.py --text "नमस्ते" --output hello.wav

# Interactive mode
python src/main.py --interactive

# Batch processing
python src/main.py --input_file texts.txt --output_dir outputs/
```