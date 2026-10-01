# 🎙️ Hey Lux — Wake Word Model (openWakeWord ONNX)

Modelo acústico de palabra de activación (**Wake Word**) entrenado para el ecosistema **Luxnode Voice Satellite** utilizando el framework open-source **[openWakeWord](https://github.com/dscripka/openWakeWord)** en Google Colab.

---

## 🎯 Frases Objetivo (Target Phrases)
El modelo fue entrenado específicamente para responder a las siguientes variaciones de voz:
- `hey lux`
- `oye lux`
- `lux`

**Nombre del Modelo:** `hey_lux`  
**Formato de Salida:** `hey_lux.onnx` (790 KB)  
**Latencia de Inferencia:** ~5 ms por frame en CPU (ONNX Runtime)  
**Tasa de Muestreo:** 16,000 Hz, 16-bit mono PCM  
**Ventana de Inferencia:** Chunks de 1280 muestras (80 ms)

---

## 🚀 Parámetros de Inferencia Recomendados
- **Umbral de Activación (`score`):** `0.35` – `0.45`
  - `0.35`: Ideal para micrófonos de escritorio a distancia natural o voces suaves.
  - `0.45`: Ideal para entornos ruidosos con música de fondo o televisión.

---

## 🧪 Cómo Probar el Modelo

1. Instalar dependencias:
```bash
pip install openwakeword onnxruntime numpy
```

2. Ejecutar prueba de verificación:
```bash
python test_model.py
```

3. Uso en Python:
```python
from openwakeword.model import Model
import numpy as np

# Cargar el modelo
oww = Model(wakeword_models=["hey_lux.onnx"], inference_framework="onnx")

# Predecir en un frame de audio de 80ms (1280 muestras int16)
audio_chunk = np.zeros(1280, dtype=np.int16)
prediction = oww.predict(audio_chunk)

print(prediction)  # {'hey_lux': 0.0}
```

---

## 📦 Integración con Luxnode
Este modelo se integra directamente con el satélite de voz local de Luxnode:
```bash
cp hey_lux.onnx e:\Luxnode\scripts\voice-satellite\models\hey_lux.onnx
```
Al iniciar `satellite_service.py`, el satélite detectará automáticamente `hey_lux.onnx` y activará la escucha 24/7 sin enviar audio a la nube.
