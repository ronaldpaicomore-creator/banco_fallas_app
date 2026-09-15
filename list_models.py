"""
list_models.py
---------------
Corre esto una sola vez para ver EXACTAMENTE qué modelos de Gemini
están disponibles para tu API key ahora mismo, en vez de adivinar
nombres que cambian con el tiempo.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("\n=== Modelos que soportan generateContent (chat / diagnóstico) ===")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

print("\n=== Modelos que soportan embedContent (embeddings / búsqueda) ===")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)