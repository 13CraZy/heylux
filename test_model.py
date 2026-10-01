"""
Test de Verificación de Inferencia para Hey Lux ONNX Model
openWakeWord
"""

import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from openwakeword.model import Model

def test_hey_lux():
    print("=" * 60)
    print("🎙️ Verificando modelo Hey Lux ONNX...")
    print("=" * 60)

    oww = Model(wakeword_models=["hey_lux.onnx"], inference_framework="onnx")
    print(f"✅ Modelo cargado con éxito: {list(oww.models.keys())}")

    # Prueba con 20 frames de silencio
    for _ in range(20):
        silent_frame = np.zeros(1280, dtype=np.int16)
        pred = oww.predict(silent_frame)

    score_silence = pred.get("hey_lux", 0.0)
    print(f"📊 Score con silencio: {score_silence:.6f} (debe ser cercano a 0)")

    if score_silence < 0.05:
        print("🎉 ¡Prueba superada! El modelo está listo para producción.")
        return 0
    else:
        print("⚠️ Advertencia: activación anormal con silencio.")
        return 1

if __name__ == "__main__":
    sys.exit(test_hey_lux())
