import torch
import numpy as np
from src.inference.tts_pipeline import SanskritTTSPipeline

# Create a test pipeline that generates a simple tone instead of using models
class TestTTSPipeline(SanskritTTSPipeline):
    def synthesize(self, text: str, output_path=None):
        # Generate a simple 440Hz tone for 1 second
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        if output_path:
            import soundfile as sf
            sf.write(output_path, audio, sample_rate)
            print(f"Test tone saved to {output_path}")
        
        return audio, sample_rate

if __name__ == "__main__":
    # Test with dummy checkpoints (they won't be loaded)
    pipeline = TestTTSPipeline(
        tacotron2_checkpoint="dummy.pth",
        hifigan_checkpoint="dummy.pth"
    )
    
    audio, sr = pipeline.synthesize("test", "test_tone.wav")
    print(f"Generated test audio: {len(audio)} samples at {sr}Hz")