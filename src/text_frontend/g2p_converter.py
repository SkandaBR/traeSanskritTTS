import re
from typing import List, Dict, Tuple

class SanskritG2PConverter:
    """Grapheme-to-Phoneme converter for Sanskrit following traditional phonetic rules."""
    
    def __init__(self):
        # Sanskrit consonant to phoneme mapping
        self.consonant_map = {
            'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
            'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
            'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
            'त': 't̪', 'थ': 't̪h', 'द': 'd̪', 'ध': 'd̪h', 'न': 'n̪',
            'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
            'य': 'j', 'र': 'r', 'ल': 'l', 'व': 'ʋ',
            'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'ɦ'
        }
        
        # Sanskrit vowel to phoneme mapping
        self.vowel_map = {
            'अ': 'ə', 'आ': 'aː', 'इ': 'ɪ', 'ई': 'iː', 'उ': 'ʊ', 'ऊ': 'uː',
            'ऋ': 'r̩', 'ॠ': 'r̩ː', 'ऌ': 'l̩', 'ॡ': 'l̩ː',
            'ए': 'eː', 'ऐ': 'əɪ', 'ओ': 'oː', 'औ': 'əʊ'
        }
        
        # Vowel diacritics (matras)
        self.matra_map = {
            'ा': 'aː', 'ि': 'ɪ', 'ी': 'iː', 'ु': 'ʊ', 'ू': 'uː',
            'ृ': 'r̩', 'ॄ': 'r̩ː', 'ॢ': 'l̩', 'ॣ': 'l̩ː',
            'े': 'eː', 'ै': 'əɪ', 'ो': 'oː', 'ौ': 'əʊ'
        }
        
        # Special symbols
        self.special_symbols = {
            'ं': 'ṃ',  # Anusvara
            'ः': 'ḥ',  # Visarga
            '्': '',   # Virama (inherent vowel killer)
            '।': '|',  # Danda
            '॥': '||' # Double danda
        }
    
    def _split_syllables(self, text: str) -> List[str]:
        """Split Sanskrit text into syllables."""
        syllables = []
        i = 0
        current_syllable = ""
        
        while i < len(text):
            char = text[i]
            
            # Handle consonants
            if char in self.consonant_map:
                if current_syllable and not current_syllable.endswith('्'):
                    syllables.append(current_syllable)
                    current_syllable = ""
                current_syllable += char
                
                # Check for virama
                if i + 1 < len(text) and text[i + 1] == '्':
                    current_syllable += text[i + 1]
                    i += 1
                else:
                    # Add inherent vowel 'a' if no explicit vowel follows
                    if i + 1 >= len(text) or text[i + 1] not in self.matra_map:
                        current_syllable += 'अ'
            
            # Handle vowels and matras
            elif char in self.vowel_map or char in self.matra_map:
                current_syllable += char
            
            # Handle special symbols
            elif char in self.special_symbols:
                current_syllable += char
            
            # Handle spaces and punctuation
            elif char.isspace() or char in '।॥':
                if current_syllable:
                    syllables.append(current_syllable)
                    current_syllable = ""
                if not char.isspace():
                    syllables.append(char)
            
            else:
                current_syllable += char
            
            i += 1
        
        if current_syllable:
            syllables.append(current_syllable)
        
        return syllables
    
    def _syllable_to_phonemes(self, syllable: str) -> List[str]:
        """Convert a single syllable to phonemes."""
        if not syllable.strip():
            return []
        
        # Handle punctuation
        if syllable in self.special_symbols:
            return [self.special_symbols[syllable]]
        
        phonemes = []
        i = 0
        
        while i < len(syllable):
            char = syllable[i]
            
            # Handle consonants
            if char in self.consonant_map:
                phonemes.append(self.consonant_map[char])
                
                # Check for virama (halant)
                if i + 1 < len(syllable) and syllable[i + 1] == '्':
                    i += 1  # Skip virama, no vowel added
                else:
                    # Check for explicit vowel (matra)
                    if i + 1 < len(syllable) and syllable[i + 1] in self.matra_map:
                        i += 1
                        phonemes.append(self.matra_map[syllable[i]])
                    else:
                        # Add inherent vowel 'a'
                        phonemes.append('ə')
            
            # Handle standalone vowels
            elif char in self.vowel_map:
                phonemes.append(self.vowel_map[char])
            
            # Handle special symbols
            elif char in self.special_symbols:
                if self.special_symbols[char]:  # Skip empty strings
                    phonemes.append(self.special_symbols[char])
            
            i += 1
        
        return phonemes
    
    def apply_sandhi_rules(self, phonemes: List[str]) -> List[str]:
        """Apply basic Sanskrit sandhi (phonetic combination) rules."""
        if len(phonemes) < 2:
            return phonemes
        
        result = []
        i = 0
        
        while i < len(phonemes):
            current = phonemes[i]
            next_phone = phonemes[i + 1] if i + 1 < len(phonemes) else None
            
            # Basic vowel sandhi rules
            if current == 'ə' and next_phone == 'ə':
                result.append('aː')  # a + a = ā
                i += 2
                continue
            elif current == 'ə' and next_phone == 'aː':
                result.append('aː')  # a + ā = ā
                i += 2
                continue
            elif current == 'aː' and next_phone == 'ə':
                result.append('aː')  # ā + a = ā
                i += 2
                continue
            
            # Consonant sandhi (simplified)
            elif current == 't̪' and next_phone and next_phone.startswith('ʃ'):
                result.append('tʃ')  # t + ś = c
                i += 2
                continue
            
            result.append(current)
            i += 1
        
        return result
    
    def convert(self, text: str) -> List[str]:
        """Convert Sanskrit text to phonemes."""
        # Split into syllables
        syllables = self._split_syllables(text)
        
        # Convert each syllable to phonemes
        all_phonemes = []
        for syllable in syllables:
            phonemes = self._syllable_to_phonemes(syllable)
            all_phonemes.extend(phonemes)
        
        # Apply sandhi rules
        phonemes = self.apply_sandhi_rules(all_phonemes)
        
        # Filter out empty phonemes
        phonemes = [p for p in phonemes if p]
        
        return phonemes
    
    def text_to_phoneme_string(self, text: str) -> str:
        """Convert text to space-separated phoneme string."""
        phonemes = self.convert(text)
        return ' '.join(phonemes)