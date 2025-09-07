import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.text_frontend.text_normalizer import SanskritTextNormalizer
from src.text_frontend.g2p_converter import SanskritG2PConverter
from src.models.tacotron2 import Tacotron2
from src.models.hifigan import Generator, HiFiGANConfig

class SanskritTTSPipeline:
    """End-to-end Sanskrit Text-to-Speech Pipeline."""
    
    def __init__(self, 
                 tacotron2_checkpoint: str,
                 hifigan_checkpoint: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Check if checkpoints exist before proceeding
        if not os.path.exists(tacotron2_checkpoint):
            raise FileNotFoundError(f"Tacotron2 checkpoint not found: {tacotron2_checkpoint}")
        if not os.path.exists(hifigan_checkpoint):
            raise FileNotFoundError(f"HiFi-GAN checkpoint not found: {hifigan_checkpoint}")
        
        # Initialize text processing components
        self.text_normalizer = SanskritTextNormalizer()
        self.g2p_converter = SanskritG2PConverter()
        self.phoneme_to_id = self._create_phoneme_mapping()
        
        # Load models
        self.tacotron2 = self._load_tacotron2(tacotron2_checkpoint)
        self.hifigan = self._load_hifigan(hifigan_checkpoint)
        
        print(f"Sanskrit TTS Pipeline initialized on {device}")
    
    def _load_tacotron2(self, checkpoint_path: str) -> Tacotron2:
        """Load Tacotron 2 model from checkpoint."""
        # Create model configuration (you may want to load this from a config file)
        class HParams:
            # Model hyperparameters
            n_symbols = 50  # Adjust based on your phoneme set
            symbols_embedding_dim = 512
            encoder_kernel_size = 5
            encoder_n_convolutions = 3
            encoder_embedding_dim = 512
            attention_rnn_dim = 1024
            attention_dim = 128
            attention_location_n_filters = 32
            attention_location_kernel_size = 31
            n_mel_channels = 80
            n_frames_per_step = 1
            decoder_rnn_dim = 1024
            prenet_dim = 256
            max_decoder_steps = 1000
            gate_threshold = 0.5
            p_attention_dropout = 0.1
            p_decoder_dropout = 0.1
            postnet_embedding_dim = 512
            postnet_kernel_size = 5
            postnet_n_convolutions = 5
            mask_padding = True
            fp16_run = False
        
        hparams = HParams()
        model = Tacotron2(hparams)
        
        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            print(f"Loading Tacotron2 checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found! Using untrained model.")
            print("This will produce noise/buzzing instead of speech.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_hifigan(self, checkpoint_path: str) -> Generator:
        """Load HiFi-GAN model from checkpoint."""
        config = HiFiGANConfig()
        model = Generator(config)
        
        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            print(f"Loading HiFi-GAN checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['generator'])
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found! Using untrained model.")
            print("This will produce noise/buzzing instead of speech.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _create_phoneme_mapping(self) -> dict:
        """Create mapping from phonemes to indices."""
        # Basic Sanskrit phoneme set - extend as needed
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
    
    def _text_to_sequence(self, text: str) -> torch.Tensor:
        """Convert text to phoneme sequence tensor."""
        # Normalize text
        normalized_text = self.text_normalizer.normalize(text)
        print(f"DEBUG: Normalized text: '{normalized_text}'")
        
        # Convert to phonemes
        phonemes = self.g2p_converter.convert(normalized_text)
        print(f"DEBUG: Generated phonemes: {phonemes}")
        
        # Convert phonemes to indices
        sequence = [self.phoneme_to_id.get('SOS', 1)]  # Start token
        skipped_phonemes = []
        
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                sequence.append(self.phoneme_to_id[phoneme])
            else:
                # Handle unknown phonemes
                skipped_phonemes.append(phoneme)
                print(f"Warning: Unknown phoneme '{phoneme}', skipping")
        
        sequence.append(self.phoneme_to_id.get('EOS', 2))  # End token
        
        if skipped_phonemes:
            print(f"DEBUG: Skipped phonemes: {skipped_phonemes}")
            print(f"DEBUG: Available phonemes: {list(self.phoneme_to_id.keys())}")
        
        print(f"DEBUG: Final sequence length: {len(sequence)}")
        return torch.LongTensor(sequence).unsqueeze(0).to(self.device)
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from Sanskrit text.
        
        Args:
            text: Sanskrit text to synthesize
            output_path: Optional path to save the audio file
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        print(f"DEBUG: Input text: '{text}'")
        
        with torch.no_grad():
            # Convert text to phoneme sequence
            sequence = self._text_to_sequence(text)
            print(f"DEBUG: Input sequence shape: {sequence.shape}")
            print(f"DEBUG: Input sequence: {sequence}")
            
            # Check if sequence is too short (only SOS + EOS tokens)
            if sequence.shape[1] <= 2:
                print("WARNING: Input sequence is too short (only start/end tokens)")
                print("This suggests phoneme mapping issues")
            
            # Generate mel-spectrogram using Tacotron 2
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.tacotron2.inference(sequence)
            print(f"DEBUG: Mel spectrogram shape: {mel_outputs_postnet.shape}")
            print(f"DEBUG: Mel spectrogram stats - min: {mel_outputs_postnet.min():.3f}, max: {mel_outputs_postnet.max():.3f}, mean: {mel_outputs_postnet.mean():.3f}")
            
            # Check for suspicious mel-spectrogram values
            if torch.isnan(mel_outputs_postnet).any():
                print("ERROR: Mel-spectrogram contains NaN values!")
            if torch.isinf(mel_outputs_postnet).any():
                print("ERROR: Mel-spectrogram contains infinite values!")
            
            # Generate audio using HiFi-GAN
            audio = self.hifigan(mel_outputs_postnet)
            audio = audio.squeeze().cpu().numpy()
            print(f"DEBUG: Raw audio shape: {audio.shape}")
            print(f"DEBUG: Audio stats - min: {audio.min():.6f}, max: {audio.max():.6f}, mean: {audio.mean():.6f}")
            print(f"DEBUG: Audio duration: {len(audio) / 22050:.3f} seconds")
            
            # Check for audio quality issues
            if np.isnan(audio).any():
                print("ERROR: Audio contains NaN values!")
            if np.isinf(audio).any():
                print("ERROR: Audio contains infinite values!")
            
            # Check if audio is just noise (high frequency content)
            audio_std = np.std(audio)
            if audio_std > 0.5:
                print(f"WARNING: Audio has high variance ({audio_std:.3f}) - likely noise/buzzing")
                print("This suggests untrained models are being used")
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            else:
                print("WARNING: Audio contains only zeros!")
            
            # Sample rate from HiFi-GAN config
            sample_rate = 22050
            
            # Save audio if path provided
            if output_path:
                sf.write(output_path, audio, sample_rate)
                print(f"Audio saved to {output_path}")
            
            return audio, sample_rate
    
    def synthesize_batch(self, texts: list, output_dir: str) -> list:
        """
        Synthesize multiple texts in batch.
        
        Args:
            texts: List of Sanskrit texts
            output_dir: Directory to save audio files
        
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"synthesis_{i:04d}.wav")
            self.synthesize(text, output_path)
            output_paths.append(output_path)
        
        return output_paths

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SanskritTTSPipeline(
        tacotron2_checkpoint="checkpoints/tacotron2_sanskrit.pth",
        hifigan_checkpoint="checkpoints/hifigan_sanskrit.pth"
    )
    
    # Test synthesis
    test_texts = [
        "नमस्ते",
        "संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति।",
        "वेदाः संस्कृते लिखिताः सन्ति।"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"Synthesizing: {text}")
        audio, sr = pipeline.synthesize(text, f"output_{i}.wav")
        print(f"Generated audio with shape: {audio.shape}, sample rate: {sr}")