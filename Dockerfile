# Dockerfile para Sistema de Trading de Criptomonedas

FROM python:3.12-slim

# Metadatos
LABEL maintainer="Trading System Team"
LABEL version="1.0"
LABEL description="Sistema Avanzado de Trading de Criptomonedas"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos
COPY requirements_avanzado.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_avanzado.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs config/secrets resultados graficos modelos

# Configurar permisos
RUN chmod +x deployment/production_system.py
RUN chmod +x lanzar_sistema.py

# Exponer puertos
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Comando por defecto
CMD ["python", "deployment/production_system.py"] 