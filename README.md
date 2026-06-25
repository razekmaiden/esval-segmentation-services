# Policía del agua - ESVAL Segmentation Service

Microservicio de segmentación semántica de imágenes satelitales para la plataforma Policía del agua - ESVAL.  
Provee una API REST (FastAPI) que combina **MobileSAM / SAM 2 Small** para segmentación de máscaras con **CLIP** para clasificación semántica de superficies (vegetación, agua, plantación, construcción).

---

## Índice

1. [Visión General](#1-visión-general)
2. [Stack Tecnológico](#2-stack-tecnológico)
3. [Arquitectura](#3-arquitectura)
4. [Modelos de Segmentación](#4-modelos-de-segmentación)
5. [Pipeline de Clasificación](#5-pipeline-de-clasificación)
6. [API Endpoints](#6-api-endpoints)
7. [Variables de Entorno](#7-variables-de-entorno)
8. [Ejecución con Docker](#8-ejecución-con-docker)
9. [Desarrollo Local](#9-desarrollo-local)
10. [Parámetros y Tuning](#10-parámetros-y-tuning)
11. [Estructura del Proyecto](#11-estructura-del-proyecto)

---

## 1. Visión General

El servicio recibe imágenes satelitales de propiedades y devuelve polígonos GeoJSON con la clasificación de cada zona superficial. Es consumido exclusivamente por el backend de `esval-web` (vía proxy en `/api/segmentation/`).

**Flujo básico**:
```
Imagen PNG (satélite) → SAM (máscaras) → CLIP (clasificación) → GeoJSON Features
```

**Clases de superficie soportadas**:
| Clase | Descripción |
|---|---|
| `water` | Piscinas, estanques, espejos de agua |
| `vegetation` | Césped, arbustos, árboles |
| `plantation` | Cultivos organizados |
| `building` | Construcciones, techos, pavimento |
| `other_surface` | Superficie no clasificada |

---

## 2. Stack Tecnológico

| Tecnología | Versión | Uso |
|---|---|---|
| **Python** | 3.11 | Runtime |
| **FastAPI** | ≥0.109.0 | Framework REST API |
| **Uvicorn** | ≥0.27.0 | ASGI server |
| **PyTorch** | 2.5.1+cu121 | Inferencia GPU/CPU |
| **torchvision** | 0.20.1+cu121 | Transforms de imagen |
| **MobileSAM** | latest (GitHub) | Segmentación ligera (9.6M params) |
| **SAM 2** | latest (GitHub) | Segmentación avanzada (46M params) |
| **CLIP (ViT-B/32)** | via `transformers` | Clasificación semántica |
| **OpenCV** | ≥4.9.0 | Procesamiento de imagen |
| **Pillow** | ≥10.2.0 | Carga/conversión de imágenes |
| **Shapely** | ≥2.0.0 | Operaciones geométricas (simplificación, validación) |
| **NumPy** | ≥1.26.0 | Operaciones matriciales |
| **Pydantic** | ≥2.6.0 | Validación de request/response |

**Hardware objetivo**: NVIDIA GPU con CUDA 12.1 (probado en RTX 3060 11GB).  
**Fallback**: CPU (funcional pero lento para imágenes grandes).

---

## 3. Arquitectura

```
app/
├── main.py          ← FastAPI app, endpoints, request/response schemas
├── segmentation.py  ← SegmentationService (singleton), carga de modelos, auto-mask
├── clip_classifier.py  ← CLIPClassifier, clasifica máscaras SAM por texto
├── morphology.py    ← Post-procesado morfológico de máscaras
├── config.py        ← Config: rutas de modelos, device detection, constantes
└── utils.py         ← Helpers: load_image, masks_to_geojson, merge_overlapping_masks
```

### Singleton de servicio

`SegmentationService` es un **singleton global** (`_service` en `segmentation.py`). Se inicializa al arrancar el servidor y puede reemplazarse en caliente via `reload_segmentation_service(engine)` sin reiniciar el proceso.

```python
service = get_segmentation_service()       # obtiene o crea singleton
reload_segmentation_service("sam2")        # hot-swap: destruye y recrea con nuevo engine
```

---

## 4. Modelos de Segmentación

### MobileSAM

| Parámetro | Valor |
|---|---|
| Arquitectura | SAM ViT-T (vit_t) |
| Parámetros | ~9.6M |
| Checkpoint | `models/mobile_sam.pt` (39 MB) |
| VRAM requerida | ~3.0 GB |
| `points_per_side` | 32 |
| `pred_iou_thresh` | 0.86 |
| `stability_score_thresh` | 0.92 |
| Tiempo por imagen | ~6,000-7,000 ms (GPU) |

**Recomendado**: Mejor detección de vegetación en imágenes satelitales (~16% cobertura vegetal detectada).

### SAM 2 Small

| Parámetro | Valor |
|---|---|
| Arquitectura | SAM 2 Hiera Small |
| Parámetros | ~46M |
| Checkpoint | `models/sam2_hiera_small.pt` (176 MB) |
| Fuente | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt` |
| VRAM requerida | ~4.0 GB |
| `points_per_side` | 32 |
| `pred_iou_thresh` | 0.80 |
| `stability_score_thresh` | 0.86 |
| Tiempo por imagen | ~5,500-6,500 ms (GPU) |

> **Nota CUDA**: SAM 2 requiere una extensión CUDA personalizada (`_C`) que puede no estar compilada en el contenedor. En ese caso se muestra una advertencia no-fatal y el post-procesado avanzado se omite. El modelo funciona correctamente.

### Modelos de checkpoint

Los checkpoints se almacenan en el volumen Docker `esval_segmentation_models` montado en `/app/models`:

```bash
/app/models/
├── mobile_sam.pt          # 39 MB
└── sam2_hiera_small.pt    # 176 MB
```

---

## 5. Pipeline de Clasificación

```
1. Recibir imagen PNG + bounds geográficos (N/S/E/W)
2. Resize si supera MAX_IMAGE_SIZE (2048px)
3. SAM genera N máscaras binarias (auto-mask generator)
4. Filtrar máscaras por tamaño mínimo (MIN_MASK_AREA = 100px²)
5. Merge de máscaras con alta superposición (IoU > umbral)
6. CLIP clasifica cada máscara:
   - Recortar region de la imagen con bounding box de la máscara
   - Comparar contra prompts de texto de cada clase
   - Asignar clase con mayor similitud coseno
7. Convertir máscaras a polígonos GeoJSON:
   - Coordenadas pixel → coordenadas geográficas (usando bounds)
   - Simplificación de contornos con Shapely
8. Devolver FeatureCollection con propiedades: zone_type, area_m2, confidence
```

### CLIP Classifier

Utiliza `openai/clip-vit-base-patch32` cargado vía HuggingFace `transformers`.  
Los prompts de texto son descriptivos en inglés para imágenes aéreas:

```python
"water": ["swimming pool", "water body", "pond", "irrigated area"]
"vegetation": ["grass lawn", "trees", "bushes", "green vegetation"]
"plantation": ["agricultural crops", "organized plantation", "cultivated fields"]
"building": ["rooftop", "concrete building", "paved surface"]
"other_surface": ["bare soil", "gravel", "mixed surface"]
```

---

## 6. API Endpoints

Base URL: `http://localhost:8001`

### `GET /health`

Estado del servicio y modelo cargado.

**Response**:
```json
{
  "status": "healthy",
  "device": { "type": "cuda", "name": "NVIDIA GeForce RTX 3060", "memory_gb": 11.76 },
  "model_loaded": true,
  "engine": "mobilesam",
  "model_variant": "mobilesam",
  "model_label": "MobileSAM"
}
```

---

### `POST /auto-segment`

Auto-segmentación completa de una imagen (sin prompts de usuario).

**Request**: `multipart/form-data`
| Campo | Tipo | Descripción |
|---|---|---|
| `image` | File (PNG/JPG) | Imagen satelital |
| `bounds` | JSON string | `{"north": f, "south": f, "east": f, "west": f}` |

**Response**:
```json
{
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [...] },
      "properties": { "zone_type": "vegetation", "area_m2": 123.4, "confidence": 0.87 }
    }
  ],
  "device_used": "cuda",
  "processing_time_ms": 6452.3
}
```

---

### `POST /segment`

Segmentación guiada con puntos de prompt o bounding box.

**Request**: `multipart/form-data`
| Campo | Tipo | Descripción |
|---|---|---|
| `image` | File (PNG/JPG) | Imagen satelital |
| `points` | JSON string | `[[x, y], ...]` en coordenadas de imagen |
| `point_labels` | JSON string | `[1, 0, ...]` (1=foreground, 0=background) |
| `bbox` | JSON string | `[x_min, y_min, x_max, y_max]` |
| `bounds` | JSON string | Bounds geográficos (opcional) |

---

### `POST /detect-region`

Detecta la región conectada en un punto del mapa (re-etiquetado asistido por IA).

**Request** (multipart/form-data):

| Campo | Tipo | Descripción |
|---|---|---|
| `image` | file | Imagen PNG del mapa |
| `bounds` | JSON string | `{north, south, east, west}` |
| `point` | JSON string | `[lat, lng]` del clic |
| `clip_geometry` | JSON string | GeoJSON geometry opcional (límite de propiedad) |

**Response**:

```json
{
  "region": {
    "type": "Polygon",
    "coordinates": [[[lng, lat], ...]]
  },
  "area_m2": 125.4,
  "current_class": "vegetation",
  "processing_time_ms": 842.1
}
```

Si no se detecta región válida, `region` es `null` y `area_m2` es `0`.

---

### `POST /switch-model`

Cambia el modelo activo en caliente (sin reiniciar el servicio).

**Request**:
```json
{ "model": "mobilesam" }
// o
{ "model": "sam2_hiera_small" }
```

**Response**:
```json
{
  "status": "success",
  "engine": "mobilesam",
  "model_label": "MobileSAM",
  "model_loaded": true,
  "message": "Modelo cambiado a MobileSAM"
}
```

> El cambio demora ~10-30s mientras el nuevo modelo se carga en VRAM.

---

### `GET /`

Información del servicio (versión, endpoints disponibles).

---

## 7. Variables de Entorno

| Variable | Default | Descripción |
|---|---|---|
| `SEGMENTATION_ENGINE` | `sam2` | Motor inicial al arrancar: `mobilesam` o `sam2` |
| `MOBILE_SAM_PATH` | `/app/models/mobile_sam.pt` | Path al checkpoint MobileSAM |
| `SAM2_MODEL_PATH` | `/app/models/sam2_hiera_small.pt` | Path al checkpoint SAM 2 |
| `HOST` | `0.0.0.0` | Host del servidor Uvicorn |
| `PORT` | `8001` | Puerto del servidor Uvicorn |

---

## 8. Ejecución con Docker

### Build

```bash
docker build -t esval-segmentation .
```

> La imagen descarga automáticamente los checkpoints durante el build (`mobile_sam.pt` y `sam2_hiera_small.pt`). Para usar un volumen externo con los checkpoints ya descargados, buildear sin el paso de descarga o montarlos como volumen.

### Ejecutar con GPU (recomendado)

```bash
docker run -d \
  --name esval-segmentation \
  --gpus all \
  -p 8001:8001 \
  --network esval_default \
  -v esval_segmentation_models:/app/models \
  -e SEGMENTATION_ENGINE=mobilesam \
  esval-segmentation
```

### Ejecutar sin GPU (CPU only)

```bash
docker run -d \
  --name esval-segmentation \
  -p 8001:8001 \
  -e SEGMENTATION_ENGINE=mobilesam \
  esval-segmentation
```

### Hot-patch del código (sin rebuild)

Para actualizar el código Python sin reconstruir la imagen:

```bash
docker cp app/main.py esval-segmentation:/app/app/main.py
docker cp app/segmentation.py esval-segmentation:/app/app/segmentation.py
docker restart esval-segmentation
```

### Verificar estado

```bash
curl http://localhost:8001/health | python3 -m json.tool
```

---

## 9. Desarrollo Local

### Requisitos

- Python 3.11+
- CUDA 12.1 (opcional, para GPU)
- ~5 GB de espacio para modelos y dependencias

### Setup

```bash
# 1. Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias (con CUDA 12.1)
pip install -r requirements.txt

# 3. Descargar checkpoints
mkdir -p models
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O models/mobile_sam.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -O models/sam2_hiera_small.pt

# 4. Iniciar servicio
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Tests

```bash
# Test básico de segmentación (requiere servicio corriendo)
python tests/test_segmentation.py
```

---

## 10. Parámetros y Tuning

### Ajuste para imágenes satelitales

Los parámetros por defecto de SAM están optimizados para fotografías naturales. Para imágenes satelitales (vista aérea cenital, resolución ~50cm/px) se aplicaron las siguientes modificaciones:

| Parámetro | SAM Default | MobileSAM (actual) | SAM 2 (actual) | Motivo |
|---|---|---|---|---|
| `points_per_side` | 32 | 32 | 32 | 48 duplica tiempo con <5% mejora |
| `pred_iou_thresh` | 0.86 | 0.86 | **0.80** | Imágenes aéreas tienen menor IoU intrínseco |
| `stability_score_thresh` | 0.92 | 0.92 | **0.86** | Texturas satelitales son menos estables |
| `min_mask_area_px` | — | 100 | 100 | Filtrar ruido (< 100px²) |

### Resultados comparativos (cliente 216783, DIGENIA 245, Reñaca)

| Métrica | MobileSAM | SAM 2 (threshold default) | SAM 2 (tuned) |
|---|---|---|---|
| Tiempo | ~6,450ms | ~5,300ms | ~5,900ms |
| Cobertura total | 369% (solapamiento) | 14.7% ❌ | 193% ✅ |
| Vegetación detectada | 16.3% ✅ | 3.0% | 1.0% ❌ |
| Agua | 2.3% | 0.7% | 1.6% |
| Plantación | 81.4% | 96.3% | 97.4% |

**Conclusión**: MobileSAM detecta mejor vegetación en imágenes satelitales (~16% vs ~1%). Recomendado como modelo por defecto.

---

## 11. Estructura del Proyecto

```
esval-segmentation-services/
├── Dockerfile               # Multi-stage: python:3.11-slim + CUDA
├── requirements.txt         # Dependencias Python (con pins CUDA 12.1)
├── README.md                # Este archivo
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + endpoints
│   ├── segmentation.py      # SegmentationService + singleton + model loading
│   ├── clip_classifier.py   # CLIPClassifier (HuggingFace ViT-B/32)
│   ├── morphology.py        # Post-procesado de máscaras
│   ├── config.py            # Config: paths, device detection, constantes
│   └── utils.py             # Helpers geoespaciales y de imagen
└── tests/
    └── test_segmentation.py # Tests de integración
```
