# Guía para Configurar el Asistente Virtual en Python

## Introducción
Esta guía proporciona los pasos necesarios para inicializar un entorno virtual en Python 3.11.0 y configurar las bibliotecas necesarias para ejecutar un asistente virtual utilizando Streamlit y otras herramientas. Es recomendable usar un entorno virtual para evitar conflictos con las versiones de las bibliotecas instaladas en tu sistema.

## Requisitos Previos
- Asegúrate de tener Python actualizado o una versión superior al 3.11.
- Familiaridad básica con la terminal o línea de comandos.

## 1. Configuración del Entorno Virtual
Primero, debes crear y activar un entorno virtual para aislar las dependencias del proyecto. 
Ejecuta los siguientes comandos en tu terminal **dentro de la carpeta** <span style="color:blue">`Asistente_Academico_`</span>:



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
```

## 3. Verificar si tus instalaciones fueron correctas

Para asegurarte de que todas las librerías se instalaron correctamente, puedes listar las bibliotecas instaladas en tu entorno virtual ejecutando el siguiente comando:

```bash
pip list
```

## 4. Ejecutar la Aplicación

Finalmente, para ejecutar la aplicación <span style="color:blue">`app.py`</span> y poner en marcha el asistente virtual, utiliza el siguiente comando:

```bash
streamlit run .\src\app.py
```

## 5. Crear archivo de configuración `.env`

Para almacenar de manera segura las credenciales de la API y el correo electrónico para enviar mensajes, crea un archivo llamado `.env` en la raíz de tu proyecto. Este archivo contendrá la clave de la API de OpenAI y la contraseña de la cuenta de correo electrónico.

1. Crea el archivo `.env` y agrega el siguiente contenido como ejemplo:

    ```plaintext
    OPENAI_API_KEY=sk-exampleAPIkey1234567890abcdef
    PASSWORD="abcd efgh ijkl mnop"
    ```

   - `OPENAI_API_KEY` es un ejemplo de la clave de acceso a la API de OpenAI, necesaria para interactuar con sus servicios.
   - `PASSWORD` es una clave de aplicación específica que permite a la cuenta de correo electrónico enviar correos.



