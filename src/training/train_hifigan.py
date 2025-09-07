import os
import time
import argparse
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

from models.hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, HiFiGANConfig

class AudioDataset(Dataset):
    """Dataset for HiFi-GAN training."""
    
    def __init__(self, data_file, config, segment_size=8192):
        self.config = config
        self.segment_size = segment_size
        
        # Load file list
        self.audio_files = []
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                for line in f:
                    audio_path = line.strip().split('|')[0]
                    if os.path.exists(audio_path):
                        self.audio_files.append(audio_path)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Ensure minimum length
            if len(audio) < self.segment_size:
                audio = np.pad(audio, (0, self.segment_size - len(audio)))
            
            # Random crop
            if len(audio) > self.segment_size:
                start = np.random.randint(0, len(audio) - self.segment_size)
                audio = audio[start:start + self.segment_size]
            
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, n_fft=1024,
                hop_length=256, win_length=1024
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return {
                'audio': torch.FloatTensor(audio),
                'mel': torch.FloatTensor(mel_spec)
            }
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return dummy data
            return {
                'audio': torch.zeros(self.segment_size),
                'mel': torch.zeros(80, self.segment_size // 256)
            }

def generator_loss(disc_outputs):
    """Generator loss."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((dg - 1) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Discriminator loss."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr - 1) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

def feature_loss(fmap_r, fmap_g):
    """Feature matching loss."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def mel_spectrogram_loss(y_true, y_pred):
    """Mel-spectrogram reconstruction loss."""
    return nn.L1Loss()(y_true, y_pred) * 45

def main():
    parser = argparse.ArgumentParser(description='Train HiFi-GAN for Sanskrit TTS')
    parser.add_argument('--config', type=str, default='config/model_config.yaml')
    parser.add_argument('--data_file', type=str, default='data/train_filelist.txt')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset and dataloader
    dataset = AudioDataset(args.data_file, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['hifigan']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    # Create models
    hifigan_config = HiFiGANConfig()
    generator = Generator(hifigan_config).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    # Optimizers
    optim_g = optim.AdamW(
        generator.parameters(),
        lr=config['hifigan']['learning_rate'],
        betas=[config['hifigan']['adam_b1'], config['hifigan']['adam_b2']]
    )
    optim_d = optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=config['hifigan']['learning_rate'],
        betas=[config['hifigan']['adam_b1'], config['hifigan']['adam_b2']]
    )
    
    # Training loop
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    for epoch in range(1000):  # Adjust as needed
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(device)
            mel = batch['mel'].to(device)
            
            # Train Generator
            optim_g.zero_grad()
            
            # Generate audio
            y_g_hat = generator(mel)
            
            # Discriminator outputs for generated audio
            y_df_hat_r, y_df_hat_g, _, _ = mpd(audio, y_g_hat.detach())
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(audio, y_g_hat.detach())
            
            # Generator loss
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            
            # Feature matching loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(audio, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(audio, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            
            # Mel loss
            loss_mel = mel_spectrogram_loss(mel, mel)  # Simplified for demo
            
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            
            loss_gen_all.backward()
            optim_g.step()
            
            # Train Discriminator
            optim_d.zero_grad()
            
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(audio, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(audio, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            
            loss_disc_all = loss_disc_s + loss_disc_f
            
            loss_disc_all.backward()
            optim_d.step()
            
            pbar.set_postfix({
                'gen_loss': loss_gen_all.item(),
                'disc_loss': loss_disc_all.item()
            })
        
        # Save checkpoint
        if epoch % 100 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'hifigan_epoch_{epoch}.pth')
            torch.save({
                'generator': generator.state_dict(),
                'mpd': mpd.state_dict(),
                'msd': msd.state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    writer.close()

if __name__ == '__main__':
    main()