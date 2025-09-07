import os
import json
import librosa
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

class SanskritDatasetPreparator:
    """Prepare Sanskrit audio-text datasets for TTS training."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.min_audio_length = 1.0  # seconds
        self.max_audio_length = 10.0  # seconds
        
    def prepare_dataset(self, 
                       audio_dir: str, 
                       transcript_file: str, 
                       output_dir: str) -> Dict:
        """Prepare dataset from audio files and transcripts."""
        
        # Load transcripts
        transcripts = self._load_transcripts(transcript_file)
        
        # Process audio files
        dataset_entries = []
        
        for audio_file, transcript in transcripts.items():
            audio_path = os.path.join(audio_dir, audio_file)
            
            if os.path.exists(audio_path):
                # Load and validate audio
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                if self._validate_audio(audio):
                    # Normalize audio
                    audio = self._normalize_audio(audio)
                    
                    # Extract features
                    mel_spectrogram = self._extract_mel_spectrogram(audio)
                    
                    # Validate transcript
                    if self._validate_transcript(transcript):
                        entry = {
                            'audio_path': audio_path,
                            'transcript': transcript,
                            'audio_length': len(audio) / self.sample_rate,
                            'mel_length': mel_spectrogram.shape[1],
                            'phonemes': self._text_to_phonemes(transcript)
                        }
                        dataset_entries.append(entry)
        
        # Save dataset metadata
        dataset_info = {
            'entries': dataset_entries,
            'total_samples': len(dataset_entries),
            'total_duration': sum(entry['audio_length'] for entry in dataset_entries),
            'sample_rate': self.sample_rate,
            'phoneme_set': self._get_phoneme_set(dataset_entries)
        }
        
        output_path = os.path.join(output_dir, 'dataset_info.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        return dataset_info
    
    def _validate_transcript(self, transcript: str) -> bool:
        """Validate Sanskrit transcript quality."""
        # Check for minimum length
        if len(transcript.strip()) < 3:
            return False
        
        # Check for proper Devanagari content
        devanagari_chars = sum(1 for char in transcript 
                              if 0x0900 <= ord(char) <= 0x097F)
        
        # At least 50% should be Devanagari
        if devanagari_chars / len(transcript) < 0.5:
            return False
        
        return True