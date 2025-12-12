# PDI-2025-2
Proyecto final de procesamiento digital de imagenes
Este proyecto implementa un modelo de detección de objetos basado en YOLO11m
entrenado sobre el dataset TACO (Trash Annotations in Context), con 58 clases de residuos.
Incluye entrenamiento en Kaggle, exportación a TorchScript, despliegue en HuggingFace Space,
y scripts de inferencia local y vía API.

- /notebooks/
   - Entrenamiento del modelo (con gráficas)
   - Exportación a TorchScript y comparación
   - Notebook de despliegue a HuggingFace Space
- /weights/
   - best.pt
   - best.torchscript
- /scripts/
   - inferencia_local.py
   - inferencia_api.py
- presentación/
   - Proyecto_final.pptx
   - 
El dataset utilizado es TACO (Trash Annotations in Context), disponible en Roboflow.
Incluye 58 clases de residuos como: botellas de plástico, cartón, latas,
envolturas, vidrio, etc.
