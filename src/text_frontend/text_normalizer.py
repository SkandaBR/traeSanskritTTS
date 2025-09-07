import re
import unicodedata
from typing import Dict, List, Tuple
from indic_transliteration import sanscript

class SanskritTextNormalizer:
    """Sanskrit text normalization for TTS preprocessing."""
    
    def __init__(self):
        self.number_words = {
            '0': 'शून्य', '1': 'एक', '2': 'द्वि', '3': 'त्रि', '4': 'चतुर्',
            '5': 'पञ्च', '6': 'षट्', '7': 'सप्त', '8': 'अष्ट', '9': 'नव',
            '10': 'दश', '11': 'एकादश', '12': 'द्वादश', '13': 'त्रयोदश',
            '14': 'चतुर्दश', '15': 'पञ्चदश', '16': 'षोडश', '17': 'सप्तदश',
            '18': 'अष्टादश', '19': 'नवदश', '20': 'विंशति'
        }
        
        self.ordinal_words = {
            '1st': 'प्रथम', '2nd': 'द्वितीय', '3rd': 'तृतीय', '4th': 'चतुर्थ',
            '5th': 'पञ्चम', '6th': 'षष्ठ', '7th': 'सप्तम', '8th': 'अष्टम'
        }
        
        self.abbreviations = {
            'श्री': 'श्रीमान्', 'डॉ': 'डॉक्टर', 'प्रो': 'प्रोफेसर',
            'etc': 'इत्यादि', 'vs': 'विरुद्ध'
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to standard form."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Convert to Devanagari if in other Indic scripts
        try:
            text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.DEVANAGARI)
        except:
            pass
        
        return text
    
    def expand_numbers(self, text: str) -> str:
        """Expand numbers to their word forms."""
        def replace_number(match):
            num = match.group()
            if num in self.number_words:
                return self.number_words[num]
            else:
                # Handle larger numbers
                return self._convert_large_number(int(num))
        
        # Replace standalone numbers
        text = re.sub(r'\b\d+\b', replace_number, text)
        return text
    
    def _convert_large_number(self, num: int) -> str:
        """Convert large numbers to Sanskrit words."""
        if num == 0:
            return 'शून्य'
        elif num < 21 and str(num) in self.number_words:
            return self.number_words[str(num)]
        elif num < 100:
            tens = num // 10
            units = num % 10
            if tens == 2:
                base = 'विंशति'
            elif tens == 3:
                base = 'त्रिंशत्'
            elif tens == 4:
                base = 'चत्वारिंशत्'
            else:
                base = f'{self.number_words.get(str(tens), str(tens))}दश'
            
            if units == 0:
                return base
            else:
                return f'{base}{self.number_words.get(str(units), str(units))}'
        else:
            # For very large numbers, return as is or implement more complex logic
            return str(num)
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
        return text
    
    def expand_symbols(self, text: str) -> str:
        """Expand symbols and punctuation."""
        symbol_map = {
            '&': 'और', '@': 'एट', '%': 'प्रतिशत', '$': 'डॉलर',
            '+': 'प्लस', '-': 'माइनस', '=': 'बराबर', '/': 'भाग'
        }
        
        for symbol, word in symbol_map.items():
            text = text.replace(symbol, f' {word} ')
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([।॥])([^\s])', r'\1 \2', text)
        text = re.sub(r'([^\s])([।॥])', r'\1 \2', text)
        
        return text.strip()
    
    def normalize(self, text: str) -> str:
        """Complete text normalization pipeline."""
        text = self.normalize_unicode(text)
        text = self.expand_numbers(text)
        text = self.expand_abbreviations(text)
        text = self.expand_symbols(text)
        text = self.normalize_punctuation(text)
        
        return text