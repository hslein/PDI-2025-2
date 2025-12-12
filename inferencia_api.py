import requests
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import sys

API_URL = "https://hslein-pdi-1.hf.space/predict"


def inferencia_api(imagen_path):
    with open(imagen_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {"image_base64": img_b64}

    print("Enviando imagen a la API...")
    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        print("Error en la API:", response.text)
        return

    data = response.json()

    print("Detecciones recibidas:")
    for d in data["detections"]:
        print(f"  - {d['class_name']} ({d['confidence']:.2f})")

    # DECODIFICAR LA IMAGEN ANOTADA (base64 → PNG)
    img_annot_b64 = data["image_annotated_base64"]
    img_bytes = base64.b64decode(img_annot_b64)
    
    # Guardar imagen anotada
    output_file = "resultado_anotado.png"
    with open(output_file, "wb") as f:
        f.write(img_bytes)

    print(f"\nImagen anotada guardada como: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python inferencia_api.py <ruta_imagen>")
    else:
        inferencia_api(sys.argv[1])
