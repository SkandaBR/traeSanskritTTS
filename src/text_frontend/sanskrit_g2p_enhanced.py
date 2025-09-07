from typing import List, Dict, Tuple
import re
from .devanagari_processor import DevanagariProcessor

class SanskritG2PEnhanced:
    """Enhanced Grapheme-to-Phoneme converter for Sanskrit with Devanagari support."""
    
    def __init__(self):
        self.devanagari_processor = DevanagariProcessor()
        
        # Phonetic rules for Sanskrit
        self.sandhi_rules = {
            # Vowel sandhi
            ('a', 'a'): 'ā',
            ('a', 'ā'): 'ā',
            ('ā', 'a'): 'ā',
            ('ā', 'ā'): 'ā',
            ('a', 'i'): 'e',
            ('a', 'ī'): 'e',
            ('ā', 'i'): 'e',
            ('ā', 'ī'): 'e',
            ('a', 'u'): 'o',
            ('a', 'ū'): 'o',
            ('ā', 'u'): 'o',
            ('ā', 'ū'): 'o',
            
            # Consonant sandhi (simplified)
            ('t', 'k'): 'tk',
            ('t', 'c'): 'cc',
            ('t', 'ṭ'): 'ṭṭ',
            ('t', 'p'): 'tp',
            ('d', 'g'): 'gg',
            ('d', 'j'): 'jj',
            ('d', 'ḍ'): 'ḍḍ',
            ('d', 'b'): 'bb',
        }
        
        # Stress patterns for Sanskrit
        self.stress_rules = {
            'penultimate_heavy': True,  # Heavy penultimate syllable gets stress
            'antepenultimate_light': True,  # Light penultimate -> stress antepenultimate
        }
    
    def syllable_to_phonemes(self, syllable: str) -> List[str]:
        """Convert a Devanagari syllable to phonemes."""
        phonemes = []
        
        # Handle special cases first
        if syllable in ['।', '॥']:
            return ['|']  # Pause marker
        
        if syllable == 'ॐ':
            return ['o', 'ṃ']
        
        # Process character by character
        i = 0
        while i < len(syllable):
            char = syllable[i]
            
            if char in self.devanagari_processor.vowels:
                phonemes.append(self.devanagari_processor.vowels[char])
                
            elif char in self.devanagari_processor.consonants:
                consonant_phoneme = self.devanagari_processor.consonants[char]
                
                # Check for virama (halant)
                if i + 1 < len(syllable) and syllable[i + 1] == '्':
                    # Consonant without inherent vowel
                    phonemes.append(consonant_phoneme[:-1])  # Remove 'a'
                    i += 1  # Skip virama
                else:
                    # Check for following matra
                    if i + 1 < len(syllable) and syllable[i + 1] in self.devanagari_processor.matras:
                        # Consonant + matra
                        base_consonant = consonant_phoneme[:-1]  # Remove inherent 'a'
                        matra_sound = self.devanagari_processor.matras[syllable[i + 1]]
                        phonemes.extend([base_consonant, matra_sound])
                        i += 1  # Skip matra
                    else:
                        # Consonant with inherent 'a'
                        phonemes.extend([consonant_phoneme[:-1], 'a'])
                        
            elif char in self.devanagari_processor.special_chars:
                special_sound = self.devanagari_processor.special_chars[char]
                if special_sound:  # Skip empty strings (like virama)
                    phonemes.append(special_sound)
            
            i += 1
        
        return phonemes
    
    def apply_sandhi_rules(self, phoneme_sequence: List[str]) -> List[str]:
        """Apply Sanskrit sandhi rules to phoneme sequence."""
        if len(phoneme_sequence) < 2:
            return phoneme_sequence
        
        result = [phoneme_sequence[0]]
        
        for i in range(1, len(phoneme_sequence)):
            prev_phoneme = result[-1]
            curr_phoneme = phoneme_sequence[i]
            
            # Check for sandhi rule
            sandhi_key = (prev_phoneme, curr_phoneme)
            if sandhi_key in self.sandhi_rules:
                # Apply sandhi: replace last phoneme with sandhi result
                result[-1] = self.sandhi_rules[sandhi_key]
            else:
                result.append(curr_phoneme)
        
        return result
    
    def add_stress_markers(self, syllables: List[str], phonemes: List[List[str]]) -> List[List[str]]:
        """Add stress markers based on Sanskrit prosody rules."""
        if len(syllables) <= 1:
            return phonemes
        
        stressed_phonemes = [p[:] for p in phonemes]  # Deep copy
        
        # Determine stress position
        stress_position = self._determine_stress_position(syllables)
        
        if 0 <= stress_position < len(stressed_phonemes):
            # Add primary stress marker
            if stressed_phonemes[stress_position]:
                stressed_phonemes[stress_position][0] = "ˈ" + stressed_phonemes[stress_position][0]
        
        return stressed_phonemes
    
    def _determine_stress_position(self, syllables: List[str]) -> int:
        """Determine stress position based on Sanskrit rules."""
        if len(syllables) <= 1:
            return 0
        
        # Check penultimate syllable weight
        penultimate_heavy = self._is_heavy_syllable(syllables[-2])
        
        if penultimate_heavy:
            return len(syllables) - 2  # Stress penultimate
        else:
            return max(0, len(syllables) - 3)  # Stress antepenultimate
    
    def _is_heavy_syllable(self, syllable: str) -> bool:
        """Determine if a syllable is heavy (guru) or light (laghu)."""
        # Heavy if: long vowel, vowel + consonant cluster, or anusvara/visarga
        long_vowels = ['ā', 'ī', 'ū', 'ṝ', 'ḹ', 'e', 'ai', 'o', 'au']
        
        # Check for long vowels
        for vowel in long_vowels:
            if vowel in syllable:
                return True
        
        # Check for anusvara or visarga
        if 'ं' in syllable or 'ः' in syllable:
            return True
        
        # Check for consonant clusters (simplified)
        consonant_count = sum(1 for char in syllable if char in self.devanagari_processor.consonants)
        if consonant_count > 1:
            return True
        
        return False
    
    def convert(self, text: str) -> List[str]:
        """Convert Sanskrit Devanagari text to phonemes."""
        # Normalize text
        normalized_text = self.devanagari_processor.normalize_text(text)
        
        # Split into syllables
        syllables = self.devanagari_processor.split_into_syllables(normalized_text)
        
        # Convert each syllable to phonemes
        all_phonemes = []
        syllable_phonemes = []
        
        for syllable in syllables:
            phonemes = self.syllable_to_phonemes(syllable)
            syllable_phonemes.append(phonemes)
            all_phonemes.extend(phonemes)
        
        # Apply sandhi rules
        all_phonemes = self.apply_sandhi_rules(all_phonemes)
        
        # Add stress markers
        stressed_syllables = self.add_stress_markers(syllables, syllable_phonemes)
        
        # Flatten with stress information
        final_phonemes = []
        for syl_phonemes in stressed_syllables:
            final_phonemes.extend(syl_phonemes)
        
        return final_phonemes