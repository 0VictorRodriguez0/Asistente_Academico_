# Guía para Configurar el Asistente Virtual en Python

## Introducción
Esta guía proporciona los pasos necesarios para inicializar un entorno virtual en Python 3.11.0 y configurar las bibliotecas necesarias para ejecutar un asistente virtual utilizando Streamlit y otras herramientas. Es recomendable usar un entorno virtual para evitar conflictos con las versiones de las bibliotecas instaladas en tu sistema.

## Requisitos Previos
- Asegúrate de tener Python actualizado o una versión superior al 3.11.
- Familiaridad básica con la terminal o línea de comandos.

## 1. Configuración del Entorno Virtual
Primero, debes crear y activar un entorno virtual para aislar las dependencias del proyecto. Ejecuta los siguientes comandos en tu terminal:

1. **Crear un entorno virtual:**
    ```bash
    python -m venv venv
    ```

2. **Si estás en PowerShell, asegúrate de permitir la ejecución de scripts:**
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

3. **Activar el entorno virtual:**
    ```bash
    .\venv\Scripts\activate
    ```

Al activar el entorno, verás que el prompt de la terminal cambia, indicando que estás trabajando dentro del entorno virtual.

## 2. Instalación de Librerías
Una vez que el entorno virtual esté activo, instala las librerías necesarias para ejecutar el asistente virtual. Es recomendable que las dependencias estén definidas en un archivo `requirements.txt`. Para instalar las librerías, ejecuta:

```bash
pip install -r requirements.txt
