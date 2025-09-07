#!/usr/bin/env python3
"""
Sanskrit Text-to-Speech Synthesizer
Main application with CLI interface
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from inference.tts_pipeline import SanskritTTSPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Sanskrit Text-to-Speech Synthesizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthesize single text
  python main.py --text "नमस्ते" --output hello.wav
  
  # Synthesize from file
  python main.py --input_file sanskrit_text.txt --output_dir ./outputs/
  
  # Interactive mode
  python main.py --interactive
        """
    )
    
    # Model arguments
    parser.add_argument('--tacotron2_checkpoint', type=str, 
                       default='checkpoints/tacotron2_sanskrit.pth',
                       help='Path to Tacotron 2 checkpoint')
    parser.add_argument('--hifigan_checkpoint', type=str,
                       default='checkpoints/hifigan_sanskrit.pth', 
                       help='Path to HiFi-GAN checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str,
                      help='Sanskrit text to synthesize')
    group.add_argument('--input_file', type=str,
                      help='File containing Sanskrit text (one sentence per line)')
    group.add_argument('--interactive', action='store_true',
                      help='Interactive mode')
    
    # Output arguments
    parser.add_argument('--output', type=str,
                       help='Output audio file path (for single text)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory (for batch processing)')
    parser.add_argument('--sample_rate', type=int, default=22050,
                       help='Audio sample rate')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Initializing Sanskrit TTS Pipeline on {device}...")
    
    # Initialize pipeline
    try:
        pipeline = SanskritTTSPipeline(
            tacotron2_checkpoint=args.tacotron2_checkpoint,
            hifigan_checkpoint=args.hifigan_checkpoint,
            device=device
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return 1
    
    # Handle different input modes
    if args.text:
        # Single text synthesis
        output_path = args.output or 'output.wav'
        print(f"Synthesizing: {args.text}")
        try:
            audio, sr = pipeline.synthesize(args.text, output_path)
            print(f"Audio saved to: {output_path}")
            print(f"Duration: {len(audio) / sr:.2f} seconds")
        except Exception as e:
            print(f"Error during synthesis: {e}")
            return 1
    
    elif args.input_file:
        # Batch processing from file
        if not os.path.exists(args.input_file):
            print(f"Input file not found: {args.input_file}")
            return 1
        
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(texts)} texts from {args.input_file}")
            output_paths = pipeline.synthesize_batch(texts, args.output_dir)
            
            print(f"Batch synthesis complete. Files saved to: {args.output_dir}")
            for i, path in enumerate(output_paths):
                print(f"  {i+1:3d}: {path}")
        
        except Exception as e:
            print(f"Error during batch processing: {e}")
            return 1
    
    elif args.interactive:
        # Interactive mode
        print("\nSanskrit TTS Interactive Mode")
        print("Enter Sanskrit text (or 'quit' to exit):")
        print("-" * 40)
        
        counter = 1
        while True:
            try:
                text = input("\n> ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                output_path = f"interactive_{counter:03d}.wav"
                print(f"Synthesizing: {text}")
                
                audio, sr = pipeline.synthesize(text, output_path)
                print(f"Audio saved to: {output_path}")
                print(f"Duration: {len(audio) / sr:.2f} seconds")
                
                counter += 1
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("\nDone!")
    return 0

if __name__ == '__main__':
    sys.exit(main())