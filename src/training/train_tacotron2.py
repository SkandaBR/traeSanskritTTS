import os
import time
import argparse
import math
import yaml
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import librosa
from tqdm import tqdm

from models.tacotron2 import Tacotron2
from text_frontend.text_normalizer import SanskritTextNormalizer
from text_frontend.g2p_converter import SanskritG2PConverter

class SanskritDataset(Dataset):
    """Dataset for Sanskrit TTS training."""
    
    def __init__(self, data_file, config):
        self.config = config
        self.text_normalizer = SanskritTextNormalizer()
        self.g2p_converter = SanskritG2PConverter()
        
        # Load data
        self.data = []
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        audio_path, text = parts[0], parts[1]
                        self.data.append((audio_path, text))
        
        # Create phoneme mapping
        self.phoneme_to_id = self._create_phoneme_mapping()
        
    def _create_phoneme_mapping(self):
        """Create mapping from phonemes to indices."""
        phonemes = [
            'PAD', 'SOS', 'EOS',  # Special tokens
            'ə', 'aː', 'ɪ', 'iː', 'ʊ', 'uː', 'r̩', 'r̩ː', 'l̩', 'l̩ː',  # Vowels
            'eː', 'əɪ', 'oː', 'əʊ',  # Diphthongs
            'k', 'kh', 'g', 'gh', 'ng',  # Velars
            'ch', 'chh', 'j', 'jh', 'ny',  # Palatals
            't', 'th', 'd', 'dh', 'n',  # Retroflexes
            't̪', 't̪h', 'd̪', 'd̪h', 'n̪',  # Dentals
            'p', 'ph', 'b', 'bh', 'm',  # Labials
            'j', 'r', 'l', 'ʋ',  # Semivowels
            'ʃ', 'ʂ', 's', 'ɦ',  # Fricatives
            'ṃ', 'ḥ',  # Anusvara, Visarga
            '|', '||',  # Punctuation
            ' '  # Space
        ]
        return {phoneme: idx for idx, phoneme in enumerate(phonemes)}
    
    def _text_to_sequence(self, text):
        """Convert text to phoneme sequence."""
        normalized_text = self.text_normalizer.normalize(text)
        phonemes = self.g2p_converter.convert(normalized_text)
        
        sequence = [self.phoneme_to_id.get('SOS', 1)]
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                sequence.append(self.phoneme_to_id[phoneme])
        sequence.append(self.phoneme_to_id.get('EOS', 2))
        
        return torch.LongTensor(sequence)
    
    def _load_mel_spectrogram(self, audio_path):
        """Load and compute mel-spectrogram from audio file."""
        if not os.path.exists(audio_path):
            # Return dummy mel-spectrogram for missing files
            return torch.zeros(80, 100)  # 80 mel channels, 100 frames
        
        try:
            audio, sr = librosa.load(audio_path, sr=22050)
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, n_fft=1024, 
                hop_length=256, win_length=1024
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return torch.FloatTensor(mel_spec)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(80, 100)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        
        # Convert text to sequence
        text_seq = self._text_to_sequence(text)
        
        # Load mel-spectrogram
        mel_spec = self._load_mel_spectrogram(audio_path)
        
        return {
            'text': text_seq,
            'mel': mel_spec,
            'text_length': len(text_seq),
            'mel_length': mel_spec.size(1)
        }

def collate_fn(batch):
    """Collate function for DataLoader."""
    # Sort batch by text length (descending)
    batch = sorted(batch, key=lambda x: x['text_length'], reverse=True)
    
    # Get max lengths
    max_text_len = max([item['text_length'] for item in batch])
    max_mel_len = max([item['mel_length'] for item in batch])
    
    # Pad sequences
    texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mels = torch.zeros(len(batch), 80, max_mel_len)
    text_lengths = torch.LongTensor([item['text_length'] for item in batch])
    mel_lengths = torch.LongTensor([item['mel_length'] for item in batch])
    
    for i, item in enumerate(batch):
        text_len = item['text_length']
        mel_len = item['mel_length']
        
        texts[i, :text_len] = item['text']
        mels[i, :, :mel_len] = item['mel']
    
    return {
        'text': texts,
        'mel': mels,
        'text_lengths': text_lengths,
        'mel_lengths': mel_lengths
    }

class Tacotron2Loss(nn.Module):
    """Loss function for Tacotron 2."""
    
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        
        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        
        mel_loss = self.mse_loss(mel_out, mel_target) + \
                   self.mse_loss(mel_out_postnet, mel_target)
        gate_loss = self.bce_loss(gate_out, gate_target)
        
        return mel_loss + gate_loss

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        texts = batch['text'].to(device)
        mels = batch['mel'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        
        # Create gate targets (1 for end of sequence, 0 otherwise)
        gate_targets = torch.zeros_like(mels[:, 0, :])  # [batch_size, max_mel_len]
        for i, mel_len in enumerate(mel_lengths):
            if mel_len > 0:
                gate_targets[i, mel_len-1] = 1.0
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(texts, mels, text_lengths, mel_lengths)
        
        # Calculate loss
        loss = criterion(outputs, [mels, gate_targets])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text'].to(device)
            mels = batch['mel'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            
            # Create gate targets
            gate_targets = torch.zeros_like(mels[:, 0, :])
            for i, mel_len in enumerate(mel_lengths):
                if mel_len > 0:
                    gate_targets[i, mel_len-1] = 1.0
            
            outputs = model(texts, mels, text_lengths, mel_lengths)
            loss = criterion(outputs, [mels, gate_targets])
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Train Tacotron 2 for Sanskrit TTS')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_file', type=str, default='data/train_filelist.txt',
                       help='Path to training data file')
    parser.add_argument('--val_data_file', type=str, default='data/val_filelist.txt',
                       help='Path to validation data file')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = SanskritDataset(args.data_file, config)
    val_dataset = SanskritDataset(args.val_data_file, config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model configuration
    class HParams:
        def __init__(self, config):
            tacotron_config = config['tacotron2']
            for key, value in tacotron_config.items():
                setattr(self, key, value)
    
    hparams = HParams(config)
    
    # Create model
    model = Tacotron2(hparams).to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = Tacotron2Loss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['training']['iters_per_checkpoint'] == 0:
            checkpoint_path = os.path.join(args.output_dir, f'tacotron2_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    writer.close()
    print('Training completed!')

if __name__ == '__main__':
    main()