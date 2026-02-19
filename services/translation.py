"""
Translation Service for multi-language support.
"""

from typing import Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Simple translation dictionary for agricultural terms
TRANSLATIONS = {
    'hi': {
        'healthy': 'स्वस्थ',
        'disease': 'रोग',
        'irrigation': 'सिंचाई',
        'weather': 'मौसम',
        'yield': 'उपज',
        'price': 'कीमत',
        'crop': 'फसल',
        'recommendation': 'सिफारिश',
        'treatment': 'उपचार',
        'plant': 'पौधा',
        'leaf': 'पत्ता',
        'soil': 'मिट्टी',
        'water': 'पानी',
        'farmer': 'किसान',
        'field': 'खेत',
        'harvest': 'कटाई',
        'Apply fungicide': 'फफूंदनाशक का प्रयोग करें',
        'Water crops': 'फसलों को पानी दें',
        'Remove infected leaves': 'संक्रमित पत्ते हटाएं'
    },
    'te': {
        'healthy': 'ఆరోగ్యం',
        'disease': 'రోగం',
        'irrigation': ' నీటిపారుమత్వం',
        'weather': ' వాతావరణం',
        'yield': 'ఇచ్చాపు',
        'price': ' ధర',
        'crop': ' పైరు',
        'recommendation': 'సిఫార్సు',
        'treatment': ' చికిత్స',
        'plant': ' చెట్టు',
        'leaf': ' ఆకు',
        'soil': ' నేల',
        'water': ' నీరు',
        'farmer': 'రైతు',
        'field': 'పొలం',
        'harvest': 'కోత'
    },
    'ta': {
        'healthy': 'ஆரோக்கியம்',
        'disease': 'நோய்',
        'irrigation': 'நீர்ப்பாசனம்',
        'weather': 'வானம்',
        'yield': 'விளைச்சல்',
        'price': 'விலை',
        'crop': 'பயிர்',
        'recommendation': 'பரிந்துரை',
        'treatment': 'சிகிச்சை',
        'plant': 'தாவரம்',
        'leaf': 'இலை',
        'soil': 'மண்',
        'water': 'தண்ணீர்',
        'farmer': 'விவசாயி',
        'field': 'வயல்',
        'harvest': 'அறுவடை'
    }
}

def translate_text(text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
    """
    Translate text to target language.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code (optional)
        
    Returns:
        Translated text
    """
    if target_lang == 'en':
        return text
    
    if target_lang not in TRANSLATIONS:
        logger.warning(f"Language {target_lang} not supported, returning original text")
        return text
    
    # Simple word-by-word translation
    words = text.split()
    translated_words = []
    
    for word in words:
        # Check for exact match
        if word.lower() in TRANSLATIONS[target_lang]:
            translated = TRANSLATIONS[target_lang][word.lower()]
        else:
            # Keep original word
            translated = word
        translated_words.append(translated)
    
    return ' '.join(translated_words)

def get_supported_languages() -> Dict[str, str]:
    """Get supported languages."""
    return {
        'en': 'English',
        'hi': 'Hindi (हिन्दी)',
        'te': 'Telugu (తెలుగు)',
        'ta': 'Tamil (தமிழ்)',
        'mr': 'Marathi (मराठी)',
        'bn': 'Bengali (বাংলা)',
        'gu': 'Gujarati (ગુજરાતી)',
        'kn': 'Kannada (ಕನ್ನಡ)',
        'ml': 'Malayalam (മലയാളം)',
        'pa': 'Punjabi (ਪੰਜਾਬੀ)'
    }

def translate_agricultural_terms(term: str, target_lang: str) -> str:
    """
    Translate single agricultural term.
    
    Args:
        term: Term to translate
        target_lang: Target language code
        
    Returns:
        Translated term
    """
    if target_lang in TRANSLATIONS:
        return TRANSLATIONS[target_lang].get(term.lower(), term)
    return term

def detect_language(text: str) -> str:
    """
    Detect language of text (simple heuristic).
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected language code
    """
    # Simple detection based on character ranges
    if any('\u0900' <= c <= '\u097F' for c in text):
        return 'hi'  # Hindi
    elif any('\u0C00' <= c <= '\u0C7F' for c in text):
        return 'te'  # Telugu
    elif any('\u0B80' <= c <= '\u0BFF' for c in text):
        return 'ta'  # Tamil
    return 'en'
