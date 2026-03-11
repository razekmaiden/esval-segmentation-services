# ESVAL Segmentation Service

Microservicio de segmentación de imágenes satelitales usando MobileSAM.

## Requisitos

- Docker con soporte GPU (opcional, fallback a CPU)
- NVIDIA Container Toolkit (para GPU)

## Uso

```bash
# Construir imagen
docker build -t esval-segmentation .

# Ejecutar con GPU
docker run -d --gpus all -p 8001:8000 esval-segmentation

# Ejecutar sin GPU (CPU only)
docker run -d -p 8001:8000 esval-segmentation
```

## Endpoints

- `GET /health` - Estado del servicio
- `POST /segment` - Segmentación con prompts
- `POST /auto-segment` - Auto-segmentación completa

## Desarrollo local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
