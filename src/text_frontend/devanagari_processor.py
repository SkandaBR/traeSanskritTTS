import unicodedata
import re
from typing import List, Tuple

class DevanagariProcessor:
    """Specialized processor for Devanagari script in Sanskrit."""
    
    def __init__(self):
        # Devanagari Unicode ranges
        self.devanagari_range = (0x0900, 0x097F)  # Main Devanagari block
        self.vedic_range = (0x1CD0, 0x1CFF)      # Vedic extensions
        
        # Character categories
        self.vowels = {
            'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
            'ऋ': 'ṛ', 'ॠ': 'ṝ', 'ऌ': 'ḷ', 'ॡ': 'ḹ', 'ए': 'e', 'ऐ': 'ai',
            'ओ': 'o', 'औ': 'au'
        }
        
        self.consonants = {
            'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'ṅa',
            'च': 'ca', 'छ': 'cha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'ña',
            'ट': 'ṭa', 'ठ': 'ṭha', 'ड': 'ḍa', 'ढ': 'ḍha', 'ण': 'ṇa',
            'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
            'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
            'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va',
            'श': 'śa', 'ष': 'ṣa', 'स': 'sa', 'ह': 'ha'
        }
        
        self.matras = {
            'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū',
            'ृ': 'ṛ', 'ॄ': 'ṝ', 'ॢ': 'ḷ', 'ॣ': 'ḹ',
            'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au'
        }
        
        self.special_chars = {
            'ं': 'ṃ',     # Anusvara
            'ः': 'ḥ',     # Visarga
            '्': '',      # Virama (halant)
            'ॐ': 'oṃ',   # Om symbol
            '।': '|',     # Danda (sentence separator)
            '॥': '||'     # Double danda
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Devanagari text using Unicode normalization."""
        # Use NFC normalization for consistent character representation
        normalized = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters that might interfere
        normalized = re.sub(r'[\u200B-\u200D\uFEFF]', '', normalized)
        
        return normalized
    
    def is_devanagari_char(self, char: str) -> bool:
        """Check if character belongs to Devanagari script."""
        code_point = ord(char)
        return (self.devanagari_range[0] <= code_point <= self.devanagari_range[1] or
                self.vedic_range[0] <= code_point <= self.vedic_range[1])
    
    def split_into_syllables(self, text: str) -> List[str]:
        """Split Sanskrit text into syllables following Devanagari rules."""
        syllables = []
        current_syllable = ""
        
        i = 0
        while i < len(text):
            char = text[i]
            
            if char in self.vowels:
                # Independent vowel starts new syllable
                if current_syllable:
                    syllables.append(current_syllable)
                current_syllable = char
                
            elif char in self.consonants:
                # Consonant handling
                if current_syllable:
                    # Check if previous syllable ends with virama
                    if current_syllable.endswith('्'):
                        # Conjunct consonant - add to current syllable
                        current_syllable += char
                    else:
                        # New syllable
                        syllables.append(current_syllable)
                        current_syllable = char
                else:
                    current_syllable = char
                    
            elif char in self.matras or char in self.special_chars:
                # Dependent vowel or special character
                current_syllable += char
                
            else:
                # Non-Devanagari character (space, punctuation, etc.)
                if current_syllable:
                    syllables.append(current_syllable)
                    current_syllable = ""
                if not char.isspace():
                    syllables.append(char)
            
            i += 1
        
        if current_syllable:
            syllables.append(current_syllable)
        
        return syllables