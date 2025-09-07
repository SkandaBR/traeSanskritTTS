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
        """
        Initialize the TTS pipeline.
        
        Args:
            tacotron2_checkpoint: Path to Tacotron 2 model checkpoint
            hifigan_checkpoint: Path to HiFi-GAN model checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Initialize text processing components
        self.text_normalizer = SanskritTextNormalizer()
        self.g2p_converter = SanskritG2PConverter()
        
        # Load models
        self.tacotron2 = self._load_tacotron2(tacotron2_checkpoint)
        self.hifigan = self._load_hifigan(hifigan_checkpoint)
        
        # Create phoneme to index mapping
        self.phoneme_to_id = self._create_phoneme_mapping()
        
        print(f"Sanskrit TTS Pipeline initialized on {device}")
    
    def _load_tacotron2(self, checkpoint_path: str) -> Tacotron2:
        """Load Tacotron 2 model from checkpoint."""
        # Create model configuration (you may want to load this from a config file)
        class HParams:
            def __init__(self):
                self.n_symbols = 150  # Adjust based on your phoneme set
                self.symbols_embedding_dim = 512
                self.encoder_kernel_size = 5
                self.encoder_n_convolutions = 3
                self.encoder_embedding_dim = 512
                self.attention_rnn_dim = 1024
                self.attention_dim = 128
                self.attention_location_n_filters = 32
                self.attention_location_kernel_size = 31
                self.n_frames_per_step = 1
                self.decoder_rnn_dim = 1024
                self.prenet_dim = 256
                self.max_decoder_steps = 1000
                self.gate_threshold = 0.5
                self.p_attention_dropout = 0.1
                self.p_decoder_dropout = 0.1
                self.postnet_embedding_dim = 512
                self.postnet_kernel_size = 5
                self.postnet_n_convolutions = 5
                self.n_mel_channels = 80
                self.mask_padding = True
                self.fp16_run = False
        
        hparams = HParams()
        model = Tacotron2(hparams).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded Tacotron 2 from {checkpoint_path}")
        else:
            print(f"Warning: Tacotron 2 checkpoint not found at {checkpoint_path}")
            print("Using randomly initialized model")
        
        model.eval()
        return model
    
    def _load_hifigan(self, checkpoint_path: str) -> Generator:
        """Load HiFi-GAN model from checkpoint."""
        config = HiFiGANConfig()
        model = Generator(config).to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['generator'])
            model.remove_weight_norm()
            print(f"Loaded HiFi-GAN from {checkpoint_path}")
        else:
            print(f"Warning: HiFi-GAN checkpoint not found at {checkpoint_path}")
            print("Using randomly initialized model")
        
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
        
        # Convert to phonemes
        phonemes = self.g2p_converter.convert(normalized_text)
        
        # Convert phonemes to indices
        sequence = [self.phoneme_to_id.get('SOS', 1)]  # Start token
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                sequence.append(self.phoneme_to_id[phoneme])
            else:
                # Handle unknown phonemes
                print(f"Warning: Unknown phoneme '{phoneme}', skipping")
        sequence.append(self.phoneme_to_id.get('EOS', 2))  # End token
        
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
        with torch.no_grad():
            # Convert text to phoneme sequence
            sequence = self._text_to_sequence(text)
            
            # Generate mel-spectrogram using Tacotron 2
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.tacotron2.inference(sequence)
            
            # Generate audio using HiFi-GAN
            audio = self.hifigan(mel_outputs_postnet)
            audio = audio.squeeze().cpu().numpy()
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
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