# Repositorio espejo — Policía del agua ESVAL

Copia de despliegue para replicación independiente de la plataforma. Contiene el código necesario para levantar el stack completo en RunPod o entorno equivalente.

| Aspecto | Detalle |
|---------|---------|
| **Organización** | [Innervycs](https://github.com/Innervycs) |
| **Rama** | `main` (única rama; sin historial previo) |
| **Origen interno** | Rama `runpod-implementation` del repositorio de desarrollo |

## Stack (tres repositorios)

| Componente | Repositorio |
|------------|-------------|
| Aplicación web (Next.js) | `Innervycs/esval-web_mirror` |
| Segmentación IA (SAM) | `Innervycs/esval-segmentation-services_mirror` |
| API SII (polígonos) | `Innervycs/esval-sii-api_mirror` |

## Despliegue

Siga la guía en `docs/operations/RUNPOD.md` (en `esval-web_mirror`). Los clones usan la rama `main` de cada repositorio `_mirror`.

## Licencia

MIT — ver archivo `LICENSE`.
