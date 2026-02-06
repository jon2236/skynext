# SkyNext AI con Llama 3.1 8B

SkyNext AI esta compuesto por FastAPI + Llama-3.1-8B-Instruct optimizado para GPUs con **8 GB VRAM** (RTX 3060/4060/etc.).

## Requisitos mínimos
- GPU NVIDIA con **≥8 GB VRAM** (recomendado 8–12 GB)
- CUDA 11.8 o 12.1+ instalado
- Python 3.10 o 3.11
- ~20–25 GB de disco libre (para descargar el modelo la primera vez)
- Git instalado

## Instalación rápida (paso a paso)

1. Clona el repo
    ```bash
    git clone https://github.com/jon2236/skynext.git
    cd skynext

2. Crea y activa un entorno virtual (muy recomendado)
    Bash# Linux / macOS
    python -m venv venv
    source venv/bin/activate

    # Windows (PowerShell o CMD)
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    (Tu prompt cambiará a (venv) PS ...)

3. Instala las dependencias
    pip install -r requirements.txt
    (La primera vez tardará unos minutos porque descarga torch, transformers, bitsandbytes, etc.)

4. Ejecuta el servidor
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

5. Abre tu navegador
    http://localhost:8000 vas a ver una interfaz de chat simple. ¡Empezá a preguntar sin limites!


## Notas importantes

La primera ejecución puede tardar 2–10 minutos: descarga ~5–6 GB del modelo cuantizado + compilación de kernels de bitsandbytes.
Si ves error de VRAM: baja max_new_tokens en app.py (ej: de 900 a 400) o cierra otros programas que usen la GPU.
Velocidad esperada: ~20–40 tokens/segundo en RTX 3060/4060 8GB.
El modelo requiere aceptar la licencia de Meta en Hugging Face (te pedirá login/token la primera vez que lo descargue).
Para detener el servidor: Ctrl + C en la terminal.

¿Querés contribuir o mejorar?
Reportá issues si algo no funciona.