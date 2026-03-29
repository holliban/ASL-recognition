from deep_translator import GoogleTranslator
import nltk
nltk.download('words', quiet=True)
from nltk.corpus import words as nltk_words
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

english_words = sorted(set(w.lower() for w in nltk_words.words()))

def get_top3(pred, class_names):
    top3_indices = pred[0].argsort()[-3:][::-1]
    return [(class_names[i], round(float(pred[0][i]) * 100, 1)) for i in top3_indices]

def get_word_suggestions(subtitle_text):
    parts = subtitle_text.split(" ")
    current_word = parts[-1].lower().strip()
    if len(current_word) < 2:
        return []
    suggestions = [w for w in english_words if w.startswith(current_word) and w != current_word]
    suggestions.sort(key=len)
    return suggestions[:5]

def translate_to_ukrainian(text):
    if not text.strip():
        return ""
    try:
        return GoogleTranslator(source='auto', target='uk').translate(text)
    except Exception:
        return "[translation failed]"

def put_unicode_text(frame, text, pos, font_size=22, color=(100, 255, 100)):
    """Draw Unicode/Cyrillic text on an OpenCV frame using Pillow."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

