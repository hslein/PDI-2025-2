from ultralytics import YOLO
import cv2
import sys
import os
import matplotlib.pyplot as plt

model = YOLO("best.pt")  # Cambia a tu nombre real

# Leer ruta de imagen desde consola
# if len(sys.argv) < 2:
#     print("Uso: py inferencia_local.py ruta_imagen.jpg")
#     sys.exit()

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print("La imagen no existe:", image_path)
    sys.exit()

# Inferencia
results = model(image_path)

# Obtener imagen original
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Dibujar predicciones
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Dibujar caja
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Etiqueta
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")

# Guardar la imagen
output_path = "resultado_inferencia.jpg"
plt.savefig(output_path, bbox_inches="tight")
print(f"\n Imagen guardada como: {output_path}")

# Mostrar detecciones en texto
print("\nDetecciones:")
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"- Clase: {cls_id} | Confianza: {conf:.2f}")