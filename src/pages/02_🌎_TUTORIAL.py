# Importaciones:
import streamlit as st
from dotenv import load_dotenv
import os
import openai
import pandas as pd
import io
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getattr
#nuevas librerias 30/10/2024
from datetime import datetime
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser

from langchain_experimental.utilities import PythonREPL
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools

from langchain.agents import create_openai_tools_agent

from langchain_openai import OpenAI
import os
from openai import OpenAI
#import seaborn as sns
import matplotlib.pyplot as plt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from email.message import EmailMessage
import ssl
import smtplib
import time
#Nueva libreria:
import unicodedata
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
    
from langgraph.prebuilt import ToolNode
import requests

            
def asistente(consulta_usuario):
    
    #Variables globales
    fecha_prueba = datetime.now().date()
    filtro_df_prueba = pd.DataFrame() #dataframe global que guarda los filtros de consulta_df
    mensaje_creado_prueba = "" #str global que guarda el mensaje creado por create_email
    asunto_creado_prueba = "" #str global que guarda el asunto creado por create_email
    #iniciar clave hola si no existe
    if "filtro_df_prueba" not in st.session_state:
        st.session_state["filtro_df_prueba"] = pd.DataFrame()   # Puedes cambiar el valor inicial
    if "mensaje_creado_prueba" not in st.session_state:
        st.session_state["mensaje_creado_prueba"] = ""  # Puedes cambiar el valor inicial
    if "asunto_creado_prueba" not in st.session_state:
        st.session_state["asunto_creado_prueba"] = ""  # Puedes cambiar el valor inicial
    if "res_prueba" not in st.session_state:
        st.session_state["res_prueba"] = None  # Puedes cambiar el valor inicial

    # Inicializar estado de la sesión 'confirmar' en 'no' si no existe
    if 'confirmar_prueba' not in st.session_state:
        st.session_state['confirmar_prueba'] = 'no'

    #bandera para confirmar escenarios
    if 'bandera_prueba' not in st.session_state:
        st.session_state['bandera_prueba'] = False
        
    #bandera para confirmar escenarios
    if 'file_prueba' not in st.session_state:
        st.session_state["file_prueba"] = False

    #fecha para obtner la fcha actual del día
    def get_current_date():
        return datetime.now().date()

    # Función para eliminar acentos de una cadena o una columna de DataFrame
    def remove_accents(data):
        if isinstance(data, pd.Series):  # Si es una columna de un DataFrame
            return data.apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn') if isinstance(x, str) else x)
        elif isinstance(data, str):  # Si es una cadena individual
            return ''.join(c for c in unicodedata.normalize('NFD', data) if unicodedata.category(c) != 'Mn')
        else:
            raise ValueError("El argumento debe ser un string o una columna de DataFrame (pandas Series)")

        
    def preprocesar_codigo(codigo):
        lineas = codigo.strip().split('\n')
        
        if not lineas[-1].strip().startswith('res ='):
            if lineas[-1].strip().startswith('return'):
                lineas[-1] = lineas[-1].replace('return', 'res =')
            else:
                lineas.append(f'res = {lineas[-1]}')

        codigo_preprocesado = '\n'.join(lineas)
        return codigo_preprocesado

    @tool
    def consulta_df(codigo):
        """
        Esta función recibe código de Python para realizar diferentes consultas sobre el 
        DataFrame global 'df'. Utiliza diversas funcionalidades de la biblioteca Pandas, 
        permitiendo a los usuarios realizar un análisis de datos de manera efectiva. 
        El código se ejecuta en un entorno seguro, empleando la librería RestrictedPython, 
        lo que previene la ejecución de código malicioso y garantiza la integridad del sistema. 
        Esta función está diseñada para ser flexible y accesible, permitiendo a los usuarios 
        realizar operaciones personalizadas adaptadas a sus necesidades específicas.
        Manda igual el df, o consultas completas para que el usuario pueda observarlas.
        """
        
        try:
            # resultado = df[(df['asignatura'].str.contains(remove_accents('Propedeutico de Matematicas'), case=False) | df['asignatura'].str.contains(remove_accents('Calculo Diferencial'), case=False)) & (df['año'] < fecha.year - 2)]
            # print(resultado)
            # Mostrar resultados
            codigo_preprocesado = preprocesar_codigo(codigo)
            # Compilar el código de forma restringida
            byte_code = compile_restricted(codigo_preprocesado, '<string>', 'exec')
            
            # Ejecutar el código
            exec(byte_code, namespace)
            # Acceder a los resultados
            resultado = namespace.get('res')
            #guardar res 
            st.session_state["res_prueba"] = resultado

            # Verificar si el resultado es un DataFrame o Series
            if isinstance(resultado, (pd.DataFrame, pd.Series)):

                st.dataframe(resultado)
                st.session_state["filtro_df_prueba"] = pd.DataFrame()  # Vaciar el dataframe
                
                try:
                    #agarrar los indices unicos de la columna matricula
                    indices  = resultado.index
                    nuevo_resultado = df.loc[indices]
                except:
                    #si falla significa que no son filtros de df y solo return resultado
                    return resultado
                
                # Eliminar duplicados de la columna 'matricula'
                alumnos_unicos = nuevo_resultado.drop_duplicates(subset='matricula', keep='first')
                st.session_state["filtro_df_prueba"] = alumnos_unicos.copy()
            return resultado
        except KeyError as e:
            return f"Error: La columna {str(e)} no se encuentra en el DataFrame."
        except Exception as e:
            # print("Error al ejecutar código:")
            # print(traceback.format_exc())
            
            # Muestra el estado de `namespace` para depurar
            # print("Estado actual de `namespace`:", namespace)
            print(f"Error al ejecutar código: {e.__class__.__name__}: {e}")
            return "No fue posible ejecutar el código!!"
        

    #Creacion de graficas
    @tool
    def create_graph(code: str):
        """
        Si te pide una variable res, hazla para que funcione todo bien.
        Especialmente diseñado para generar gráficos a partir de consultas sobre la tabla de datos df 
        o una tabla llamada resultados. Si el código incluye la creación de gráficos, asegúrate de importar 
        las bibliotecas necesarias, como matplotlib, y de llamar a plt.show() al final para visualizar el gráfico. 
        Además, ajusta automáticamente el tamaño y la configuración para cualquier tipo de gráfico, y guarda 
        las gráficas en el estado de la sesión.
        """

        res =  st.session_state["res_prueba"] 
        #print("indices", res.shape)
        # Asegura que realice graficas
        if "plt." in code and "plt.show()" not in code:
            code += "\nplt.show()"
        code = code.replace("plt.show()", "")
        code += "\nbuf = io.BytesIO()\nplt.savefig(buf, format='png',bbox_inches='tight')\nbuf.seek(0)\nst.session_state['current_graph_prueba'] = buf"

        try:
            exec(code)
            # st.session_state["res"] = None
            return {"output": "Gráfica generada."}  # Asegúrate de devolver algo si es necesario
        except Exception as e:
            st.error(f"Error al ejecutar el código: {e}")
            return {"output": str(e)}  # Retorna el error también
        
    # Para envios de correos electronicos
    @tool
    def create_email(mensaje,asunto):
        """
            Esta función recibe mensaje y el asunto del mensaje,
            Esta función crea un mensaje solo cuando el usuario lo solicite con saltos de linea y triple("),
            Esta función crea el asunto especificado por el usuario.
            El mensaje debe contar con las especificaciones del usuario que puede tener el nombre de las columnas del dataframe df
            columnas de la varible 'df' los datos de su matricula,p1,p2,p3,alumno,clave_asig,asignatura,seccion,periodo,num_docente,docente.
            Considera que el mensaje debe tener un formato y debe incluir saltos de línea para que sea más legible.

        """ 
        print("entra a bandera",st.session_state["bandera_prueba"])

        # global mensaje_creado, asunto_creado
        st.session_state["mensaje_creado_prueba"] = mensaje
        st.session_state["asunto_creado_prueba"] = asunto

        #Unir el mensaje con su asunto
        mensaje_con_asunto = asunto + "\n\n" + mensaje
        st.session_state["bandera_prueba"] = True 

        print("Salida", st.session_state["confirmar_prueba"])
        print("sale",st.session_state["bandera_prueba"])
        return mensaje_con_asunto

    def send_email():
        print("Entro PAPUS")
        filtro_df_prueba = st.session_state["filtro_df_prueba"]
        mensaje_creado_prueba = st.session_state["mensaje_creado_prueba"]
        asunto_creado_prueba = st.session_state["asunto_creado_prueba"]

        if filtro_df_prueba.empty and mensaje_creado_prueba == "" and asunto_creado == "":
            st.warning("No se encuentran datos del mensaje y de los alumnos")
            st.session_state["confirmar_prueba"] = None
            if st.button("Reintentar"):
                st.session_state["bandera"] = False
                st.experimental_rerun()  # Reinicia la aplicación
        elif filtro_df_prueba.empty or mensaje_creado_prueba == "" or asunto_creado == "":
            vacias = []
            
            if filtro_df_prueba.empty:
                vacias.append("datos de alumnos")
            if mensaje_creado_prueba == "":
                vacias.append("mensaje")
            if asunto_creado_prueba == "":
                vacias.append("asunto del mensaje")
            
            st.warning(f"No se encuentran los siguientes datos: {', '.join(vacias)}")
            st.session_state["confirmar"] = None
            if st.button("Reintentar"):
                st.session_state["bandera"] = False
                st.experimental_rerun()  # Reinicia la aplicación
        else:
            #restablecer el estado confirmar, para no seguir enviando mensajes
            st.session_state['confirmar'] = None

            # proceso para enviar el mensaje
            password = os.getenv("PASSWORD")
            email_sender = "ceprintdo@gmail.com"

            context = ssl.create_default_context()
            print("Buenas"+ filtro_df_prueba["alumno"]) 
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                smtp.login(email_sender, password)
                for _, row in filtro_df_prueba.iterrows():
                    email_reciver = str(row['matricula']) + "@ucaribe.edu.mx"  # Convertir la matrícula a cadena
                    body = mensaje_creado_prueba.format(alumno=row['alumno'], asignatura=row['asignatura'], p1=row['p1'],
                                                    p2=row['p2'],p3=row['p3'],final=row['final'],clave_asig=row['clave_asig'],
                                                    seccion=row['seccion'],periodo=row['periodo'],
                                                    num_docente=row['num_docente'],docente=row['docente'])
                    em = EmailMessage()
                    em["From"] = email_sender
                    em["To"] = email_reciver
                    em["Subject"] = asunto_creado_prueba
                    em.set_content(body)
                    smtp.sendmail(email_sender, email_reciver, em.as_string())
            # Para vaciar las variables nuevamente:
            st.session_state["filtro_df_prueba"] = pd.DataFrame()    # Vaciar el DataFrame
            st.session_state["mensaje_creado_prueba"] = ""  # Vaciar el string
            st.session_state["asunto_creado_prueba"] = ""  # Vaciar el string
            st.success("mensaje enviado con exito") 
            st.session_state["confirmar_prueba"] = None
            if st.button("Finalizar"):
                st.session_state["bandera_prueba"] = False
                st.experimental_rerun()  # Reinicia la aplicación
                
    # Función para manejar la carga de archivos CSV
    def cargar_archivo_csv():
        url = "https://raw.githubusercontent.com/0VictorRodriguez0/AsistenteAcademico/main/datos_simulados.csv"
        response = requests.get(url)
        
        if response.status_code == 200:
            df = pd.read_csv(url)
            
            df['año'] = df['periodo'].str[:4]
            df['año'] = df['año'].astype(int) 
            #agregar periodo_int
            df['periodo_int'] = df['periodo'].str[:6]
            df['periodo_int'] = df['periodo_int'].astype(int)  
            #quitar acentos
            df['asignatura'] = remove_accents(df['asignatura'])
            st.session_state.df = df
            
            st.session_state["file_uploaded_prueba"] = True
            st.session_state["df_prueba"] = df
            return df
        else:
            st.error("No se pudo cargar el archivo desde GitHub. Verifica la URL o tu conexión a internet.")
            return None


    def subir_archivo(): 
        if st.button("📎", key="upload_another"):
            st.session_state["file_uploaded_prueba"] = False
            cargar_archivo_csv()
            
    def mensajes_anteriores():
        # Mostrar mensajes anteriores, junto con gráficas si existen
        for msg in st.session_state["messages_prueba"]:
            avatar_image = avatares.get(msg["role"], None)
            st.chat_message(msg["role"], avatar=avatar_image).write(msg["content"])
            
            # Si el mensaje tiene una gráfica, mostrarla
            if msg.get("graph_prueba") is not None:
                st.image(msg["graph_prueba"])

    st_callback = StreamlitCallbackHandler(st.container())

    load_dotenv()

    # Configura tu API Key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Nombre del asistente
    assistant_name = "Pedro"

    # Avatares
    avatares = {
        "assistant": "./img/coco2.png",
        "user": "./img/user2.png"
    }
    
    # Insertar CSS personalizado para efecto de resplandor
    st.markdown(
        """
        <style>
        .body{
            background: white; 
        }
        .st-emotion-cache-janbn0 {
            border: 1px solid transparent;
            padding: 10px 15px; /* Ajustamos padding para dejar espacio */
            margin: 0px 7px;
            max-width: 50%;
            margin-left: auto;

            background: #2F2F2F;
            color: white;
            border-radius: 20px;

            flex-direction: row-reverse;
            text-align: justify;
        }

        .st-emotion-cache-janbn0 p {
            margin-top: 0.5em;   /* Pequeño margen arriba */
            margin-bottom: 0.5em; /* Pequeño margen abajo */
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inicializar mensajes y estado de archivo
    if "messages_prueba" not in st.session_state:
        st.session_state["messages_prueba"] = []
          


    if "file_uploaded_prueba" not in st.session_state:
        st.session_state["file_uploaded_prueba"] = False

        
    if "df_prueba" not in st.session_state:
        st.session_state["df_prueba"] = None
        


    df = cargar_archivo_csv()


    # Si el archivo se subio comenzamos a chatear   
    if st.session_state["file_uploaded_prueba"]:
            
        df = st.session_state.get('df_prueba', None)
        # Espacio de nombres seguro
        namespace = safe_globals.copy()
        namespace['_getattr_'] = default_guarded_getattr
        namespace['_getitem_'] = default_guarded_getitem
        namespace['__builtins__'] = None  # Deshabilitar acceso a built-ins inseguros
        namespace['pd'] = pd  # Permitir uso de pandas
        namespace['df'] =  df  # Permitir acceso al DataFrame
        namespace['datetime'] = datetime  # Permitir acceso controlado a datetime
        namespace['fecha'] = fecha_prueba #permitir acceso a la variable fecha
        namespace['remove_accents'] = remove_accents 
        namespace['get_current_date'] = get_current_date

        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        tools = [consulta_df, create_graph, create_email]
        tool_node = ToolNode(tools)

        llm_chat = ChatOpenAI(temperature=0, streaming=True)
        #llm_chat = llm_chat.bind_tools(tools)

        template = """
        Eres un asistente que apoya al usuario a realizar consultas y análisis de datos académicos, gestionando información sobre calificaciones de estudiantes. 
        Además, puedes generar gráficos y enviar correos electrónicos según se requiera. La tabla de datos se identifica con la variable df y contiene las siguientes columnas:
        - 'matricula': variable categórica que identifica de forma única a un estudiante.
        - 'p1', 'p2', 'p3': variables numéricas que indican las calificaciones del estudiante en los periodos parciales de la asignatura.
        - 'final': variable numérica que indica la calificación final del estudiante en la asignatura.
        - 'alumno': variable categórica con el nombre del estudiante.
        - 'clave_asig': variable categórica que indica un ID para una asignatura.
        - 'asignatura': variable categórica con el nombre de la asignatura.
        - 'seccion': variable categórica que indica la sección de la asignatura.
        - 'periodo': variable categórica que indica el periodo lectivo.
        - 'num_docente': variable categórica que identifica al profesor de la asignatura.
        - 'docente': variable categórica con el nombre del profesor.
        - 'año': variable numérica que indica el año en el cual se curso o se esta cursando la asignatura.
        - 'periodo_int': variable numerica que indica el periodo lectivo.
        ### la  columna asignatura
        - la base de datos contiene la columna 'asignatura' la cua lcorresponde a una asignatura o materia.
        - una asignatura puede estar escritas con acentos, mayusculas y minusculas.
        - Considera que cuentas con una función definida remove_accents() que permite quitar acentos de las columnas del df y de strings, 
        Por ejemplo, si el usuario desea conocer los alumnos reprobados en los parciales 1 y 2 de cálculo, el código sería:
        df[(df['p1'] < 7) | (df['p2'] < 7)].loc[df['asignatura'].str.contains(remove_accents('Cálculo'), case=False)],
        En el ejemplo se usa remove_accents en la palabra 'Cálculo' para que la búsqueda no se vea afectada por acentos.


        ### la columna 'periodo'
        - la base de datos contiene la columna 'periodo' la cual esta conformada por "año" - "periodo del año" - "estación". 
        - ejemplo de periodo "201503", el año es "2015", el periodo del año es "03" y la estación es "otoño"
        - El periodo tiene tres estaciones o temporadas, Primavera = 01, Verano = 02, Otoño = 03.
        - El periodo igual contiene el nombre de la estación, ejemplo "202301 Primavera", contiene la estación primavera.
        - para hacer comparaciones de periodo con datos numeros utiliza la columna periodo_int, la columna contiene el año y el periodo del año,
        ejemplo: 202101, "2021" es el año y "01" el periodo del año.
        - Para conocer el periodo actual o semestre actual utiliza df['periodo_int'].max().
        - el periodo tambien se le puede referir al ciclo.
        

        ### Criterios de Evaluación
        - Un estudiante aprueba si su calificación final es mayor o igual a 7 (promedio de 'p1', 'p2', 'p3').
        - Un estudiante reprueba si su calificación final es menor a 7.
        
        ### Herramientas Disponibles:
        - **consulta_df**: para consultas de datos.
        - **create_graph**: para generar gráficos.
        - **create_email**: para escribir y enviar correos electrónicos.

        ### Instrucciones Generales:
        1. **Consultas**:
            - Para realizar consultas, asegúrate de proporcionar código claro y eficiente, usando operaciones de pandas. El resultado final de la consulta debe asignarse a 'res'.
            - Usa `remove_accents()` para búsquedas de texto que puedan incluir acentos.
            - Siempre que se busque un término específico en una columna, usa `str.contains()` con `case=False` para asegurar la búsqueda sin distinción de mayúsculas/minúsculas.
            - Cuando se necesite filtrar por periodo, usa `str.contains()` para permitir buscar patrones específicos, ejemplo: `df['periodo'].str.contains('202401', case=False)`.
            - Para listas de estudiantes o datos tabulares, muestra el resultado con `st.session_state.get("res")`.
            - Usa df['año'].max() si necesitas conocer el año actual.

        2. **Gráficos**:
            - Usa `create_graph` para generar gráficos. Indica el tipo de gráfico (barras, histogramas, cajas, series de tiempo) y asegúrate de que sea adecuado para el análisis solicitado.
            - Ejemplo: si se requiere una gráfica de barras que muestre la tasa de reprobación, usa `create_graph` para construir dicha gráfica.

        3. **Correos Electrónicos**:
            - Usa `create_email` cuando se mencione explícitamente la creación o envío de correos electrónicos, o si el usuario hace referencia a contactar a alguien.
            - Los mensajes deben ser personalizados. Por ejemplo: "Hola {{alumno}}, notamos que tu calificación en {{asignatura}} es {{final}}. Te recomendamos..."
            - Asegúrate de presentar el correo para aprobación antes de proceder con el envío.
        

        ### Ejemplos de Consultas y Casos de Uso:
        1. **Detección de estudiantes en riesgo**:
            - "Lista a los estudiantes que estén cursando alguna asignatura en el periodo 2024-01, y que hayan reprobado la misma asignatura en periodos previos; además, que en el periodo actual tengan algún parcial reprobado en esa misma asignatura."
                Código: `res = df[(df['periodo'].str.contains('202401', case=False)) & ((df['p1'] < 7) | (df['p2'] < 7) | (df['p3'] < 7)) & (df['clave_asig'].duplicated(keep=False))]`
            - "Genera una lista de estudiantes de la cohorte 2022 con una probabilidad de deserción escolar mayor a 0.3."
                Código: `res = df[(df['cohorte'] == 2022) & (df['probabilidad_desercion'] > 0.3)]`

        2. **Generación automática de reportes académicos**:
            - Personales: 
                - "Calcula el promedio de calificaciones del estudiante xxxx."
                    Código: `res = df[df['alumno'] == 'xxxx'][['p1', 'p2', 'p3']].mean(axis=1).mean()`
            - Grupales:
                - "Construye una gráfica de barras que muestre la tasa de reprobación de las 10 asignaturas con la tasa más alta."
                    Código: `res = df[df['final'] < 7].groupby('asignatura')['final'].count() / df.groupby('asignatura')['final'].count()`
                    Usa: `create_graph` para generar la gráfica de barras.

        3. **Alertas automáticas**:
            - "Identifica a los estudiantes que no aprobaron el primer parcial de cálculo vectorial en 202401 y crea un correo invitándoles a recibir apoyo."
                Realiza: `create_email` para generar el correo personalizado.

        4. **Análisis de tendencias en el rendimiento académico**:
            - "Lista las asignaturas cuya tasa de reprobación haya aumentado entre el periodo 202303 y 202403."
                Código: `res = df.groupby(['asignatura', 'periodo'])['final'].apply(lambda x: (x < 7).mean()).unstack().diff(axis=1).loc[:, '202403']`

        ### Recomendaciones para consultas
        1. **Recomendaciones para la Visualización de Datos**:
            - El asistente proporcionará recomendaciones sobre los gráficos más adecuados para las consultas del usuario. 
            Por ejemplo, si el usuario solicita información sobre el promedio de calificaciones, el asistente puede sugerir al final de la consulta las siguientes opciones de gráficos:
            Gráfico de barras, Gráfico de líneas, Histograma, Gráfico de dispersión, Gráfico de caja, Gráfico de pastel, Gráfico de áreas, Gráfico de barras apiladas, Gráfico de barras horizontales, 
            Mapa de calor, Gráfico de series temporales, Gráfico de radar, Gráfico de histogramas acumulativos, Gráfico de barras agrupadas, Gráfico de líneas múltiples.
            - En la recomendación el asistente tiene conocimiento de las graficas utilizadas en matplolib.
            - las recomendaciones siempre se escriben al final de la consulta.
            - El numero de recomendaciones de nombres de graficos se debe limitar a 3, pueden ser menos recomendaciones pero deben ser menor o igual a 3 recomendaciones.
            - las graficas recomendadas deben considerarse adecuadas a la consulta que realizo el usuario.
            - solo considera recomendar las Graficas cuanto sientas que sea necesario y creas que permitan obtener una mejor respuesta para la visualización de los datos.
        2. **Recomendaciones para la herramienta consulta_df**:
            - considera llamar varias vaces la herramienta 'consulta_df' si crees que la consulta necesita varios codigos para que sea repondida adecuadamente.
        3. **Reomendaciones de herramientas**:
            - Considera recomendar la herraminenta de 'consulta_df' o 'create_graph' si crees que la respuesta puede ser mejor para visualizarla,
            en este contexto, visualizarlo mejor con solo datos escritos o visuales con el uso de graficas. 
            
        ### Nota:
        - Para consultas complejas que impliquen cálculos detallados o varias condiciones, desglosa la lógica en pasos claros y comprensibles.
        - Asegúrate de que todos los cálculos y resultados asignen el valor final a `res` para evitar errores en la ejecución.
        - puedes utilizar todas las graficas de Matplotlib.
        - Dale consejos a los usuarios de las graficas que podrian utilizar para visualizar sus datos,
        por ejemplo si la consulta es de el promedio de calificaciones, puede dar consejos el asistente de utilizar grafica de barras,lineas, histogra,
        solo da el nombre de las graficas que creas que son mejores para la consulta.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("user", "Tu consulta actual es: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm_chat, tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        
        
        
        if prompt := consulta_usuario:
            # Chat del usuario para el asistente
            st.session_state["messages_prueba"].append({"role": "user", "content": prompt})
            st.chat_message("user", avatar=avatares["user"]).write(prompt)

            # Ejecutar la consulta y guardar la respuesta del asistente
            with st.chat_message("assistant", avatar=avatares["assistant"]):

                st_callback = StreamlitCallbackHandler(st.container())

                # Muestra todo el proceso de lenguaje natural
                response = agent_executor.invoke(
                    {"input": prompt}, {"callbacks": [st_callback]}
                )

                # Asegúrate de que la respuesta esté definida
                if 'output' in response:
                    message_content = response["output"]
                else:
                    message_content = "No se obtuvo respuesta del asistente."

                # Guardar la respuesta del asistente
                current_graph = st.session_state.get("current_graph_prueba", None)

                st.session_state["messages_prueba"].append({"role": "assistant", "content": message_content, "graph_prueba": current_graph})

                # Mostrar la respuesta del asistente
                st.write(message_content)

                if current_graph:
                    st.image(current_graph)

                # Limpiar la gráfica actual para no mostrarla en futuros mensajes
                st.session_state["current_graph_prueba"] = None  

            # Comprobar si la bandera está activada para la confirmación de envío de mensajes
            if st.session_state.get("bandera_prueba", False):
                st.write("¿Confirmar envío de mensajes?")
                
                # Botones de confirmación
                col1, col2 = st.columns([.1, .7])  # Crear dos columnas para los botones

                with col1:
                    if st.button("Sí", key="confirmar_si"):
                        send_email()  # Llamada a la función de envío de email
                        st.success("Has confirmado el envío de mensajes.")
                        st.session_state['bandera_prueba'] = False  # Resetear bandera

                with col2:
                    if st.button("No", key="confirmar_no", type="primary"):
                        st.warning("Has cancelado el envío de mensajes.")
                        st.session_state['bandera_prueba'] = False  # Resetear bandera
                st.session_state['bandera_prueba'] = False  # Resetear bandera
        
        
    else:
            st.error("Por favor, sube primero el archivo CSV para poder interactuar con el asistente.",icon="⚠️")
# Inicializar la sesión de estado para almacenar la subpágina actual
if "subpagina" not in st.session_state:
    st.session_state.subpagina = "Introducción"

with st.sidebar:
    # Menú de subpáginas en la barra lateral
    secciones = ["Introducción", "Herramientas de consultas", "Herramientas de Gráficas", "Generación automática de reportes académicos"]
    subpagina = st.sidebar.radio("Selecciona una sección:", secciones, index=secciones.index(st.session_state.subpagina))
    st.sidebar.markdown("---")

# Actualizar la sesión de estado con la selección del menú de la barra lateral
st.session_state.subpagina = subpagina

# Función para mostrar los botones de navegación personalizados
def mostrar_botones_navegacion():
    col1, col2 = st.columns([1, 0.3])

    # Botón "Anterior" solo si no estamos en la primera página
    if secciones.index(st.session_state.subpagina) > 0:
        with col1:
            if st.button("Anterior", key=f"anterior_{st.session_state.subpagina}"):
                st.session_state.subpagina = secciones[secciones.index(st.session_state.subpagina) - 1]

    # Botón "Siguiente" solo si no estamos en la última página
    if secciones.index(st.session_state.subpagina) < len(secciones) - 1:
        with col2:
            if st.button("Siguiente", key=f"siguiente_{st.session_state.subpagina}"):
                st.session_state.subpagina = secciones[secciones.index(st.session_state.subpagina) + 1]
    else:
        # Botón para ir al formulario de Google Forms en la última página
        with col2:
            st.markdown("[Completar evaluación en Google Forms](https://forms.gle/Er45LbbMyApKpPXZ8)", unsafe_allow_html=True)


# Lógica para mostrar el contenido según la subpágina seleccionada
if st.session_state.subpagina == "Introducción":
    st.write("# Introducción: ")
    st.write("""
    ### **Bienvenido al asistente virtual Pedro**  
    Este asistente está diseñado para ayudarte a consultar y analizar las calificaciones de estudiantes almacenadas en una tabla de datos usando Python con la librería pandas. Aquí tienes una lista de las funcionalidades que puedes utilizar:
    """)
    url2 = "https://raw.githubusercontent.com/0VictorRodriguez0/AsistenteAcademico/main/datos_simulados.csv"

    # Leer el archivo CSV desde la URL
    dfee = pd.read_csv(url2)

    # Seleccionar las primeras 5 filas
    df_5_filas = dfee.head(5)
    
    st.write("Mostrando las primeras 5 filas del archivo CSV de ejemplo:")
    st.dataframe(df_5_filas)
    st.write("""
    **Variables que maneja el asistente**:
    
    - 'matricula': Identificación única del estudiante.
    - 'p1', 'p2', 'p3': Calificaciones de los parciales 1, 2 y 3.
    - 'final': Calificación final (promedio de p1, p2, p3).
    - 'alumno': Nombre del estudiante.
    - 'clave_asig': ID único de la asignatura.
    - 'asignatura': Nombre de la asignatura.
    - 'seccion': Sección de la asignatura.
    - 'periodo': Periodo académico de la asignatura.
    - 'num_docente': Identificación del docente.
    - 'docente': Nombre del docente.
    
    ### Ejemplos de consultas:
    
    1. **Realizar consultas sobre los datos**:
        - Puedes pedirle al asistente que realice diferentes tipos de consultas sobre los datos de las calificaciones. 
        - Ejemplo: _"Muestra los estudiantes reprobados en la asignatura de cálculo."_
        - Ejemplo: _"¿Cuántos estudiantes han aprobado el primer parcial de ecuaciones diferenciales?"_
    
    2. **Generar gráficos**:
        - Puedes solicitarle al asistente que genere gráficos para visualizar los datos. 
        - Ejemplo: _"Genera una gráfica de barras de los alumnos reprobados y aprobados en ecuaciones diferenciales."_
        - Ejemplo: _"Genera un gráfico de pastel que muestre la proporción de estudiantes aprobados y reprobados en ecuaciones diferenciales."_

    3. **Enviar mensajes personalizados**:
        - El asistente puede generar y enviar mensajes basados en la información de las calificaciones de los estudiantes.
        - Ejemplo: _"Envía un correo a los estudiantes que reprobaron el segundo parcial de programación."_
        - Ejemplo: _"Genera un mensaje para informar a los estudiantes sus calificaciones finales en álgebra."_


    ### Ejemplos de consultas comunes:

    - _"¿Cuántos estudiantes reprobaron el primer parcial de álgebra?"_
    - _"Genera una gráfica de pastel de los estudiantes aprobados y reprobados en física."_
    - _"Envía un mensaje a los estudiantes que tienen una calificación final menor a 7 en ecuaciones diferenciales."_  

    Si necesitas más ayuda, simplemente pregunta al asistente sobre lo que quieras consultar o realizar.
    """)
    mostrar_botones_navegacion()

    
if st.session_state.subpagina == "Herramientas de consultas":
    st.write("# Herramientas de consultas: ")
    st.write("## Casos de uso: ")
    st.write("### **1-. Detección temprana de estudiantes en riesgo:**")
    
    # Contenedor para la consulta del usuario
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario = st.text_area("Ejemplo de consulta:", 
                                        value="Lista a los estudiantes que estén cursando alguna asignatura en el periodo actual, 202301, y que hayan reprobado la misma asignatura en periodos previos; además, que en el periodo actual tengan algún parcial reprobado en esa misma asignatura")
        enviar_consulta = st.button("Enviar", key="enviar_consulta")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta:
                asistente(consulta_usuario)   
    
    # Contenedor para la consulta del usuario
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario = st.text_area("Ejemplo de consulta:", 
                                        value="Muestra los promedios de las secciones de cálculo diferencial en el periodo 202301.")
        enviar_consulta = st.button("Enviar", key="enviar_consulta_4")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta:
                asistente(consulta_usuario) 
                
    # Contenedor para la consulta del usuario
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario = st.text_area("Ejemplo de consulta:", 
                                        value="¿Cuáles son las tres asignaturas con mayor tasa de reprobación?")
        enviar_consulta = st.button("Enviar", key="enviar_consulta_5")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta:
                asistente(consulta_usuario)
                
                
    st.write("### **2.	Generación automática de reportes académicos:**")
    # Contenedor para la consulta del usuario
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario_1 = st.text_area("Ejemplo de consulta:", 
                                        value="Calcula el promedio de calificaciones del estudiante de matricula GOOFOOON9")
        enviar_consulta_1 = st.button("Enviar", key="enviar_consulta_1")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta_1:
                asistente(consulta_usuario_1)  
                
    # Contenedor para la consulta del usuario
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario_1 = st.text_area("Ejemplo de consulta:", 
                                        value="¿Cuál ha sido el cambio en el promedio de cada parcial en álgebra durante el periodo 202301?")
        enviar_consulta_1 = st.button("Enviar", key="enviar_consulta_6")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta_1:
                asistente(consulta_usuario_1) 
                
    mostrar_botones_navegacion()
            
            
elif st.session_state.subpagina == "Herramientas de Gráficas":
    st.write("# Herramientas de Gráficas: ")
    st.write("## Casos de uso: ")
    st.write("### **1-. Generación automática de reportes académicos**")
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario_2 = st.text_area("Ejemplo de consulta:", 
                                        value="Construye una gráfica de barras que muestre la tasa de reprobación de las asignaturas, ordenadas descendentemente. Muestra solamente las 10 asignaturas con la tasa de reprobación más alta.")
        enviar_consulta_2 = st.button("Enviar", key="enviar_consulta_2")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta_2:
                asistente(consulta_usuario_2)
                
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario_2 = st.text_area("Ejemplo de consulta:", 
                                        value="Compara, mediante gráficas de caja paralelas, la distribución de las calificaciones en las secciones de cálculo integral en el periodo 202303. ")
        enviar_consulta_2 = st.button("Enviar", key="enviar_consulta_8")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta_2:
                asistente(consulta_usuario_2)
                
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario_2 = st.text_area("Ejemplo de consulta:", 
                                        value="Construye una serie de tiempo de la tasa de reprobación en la asignatura de ecuaciones diferenciales en cada periodo lectivo.")
        enviar_consulta_2 = st.button("Enviar", key="enviar_consulta_7")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta_2:
                asistente(consulta_usuario_2)
                
                
    st.write("### **2-. Generación automática de reportes académicos**")
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario_3 = st.text_area("Ejemplo de consulta:", 
                                        value="Compara la distribución de calificaciones, obtenidas en el periodo 202301, correspondientes a las asignaturas de cálculo diferencial, cálculo integral y ecuaciones diferenciales. Emplea gráficas de caja paralelas.")
        enviar_consulta_3 = st.button("Enviar", key="enviar_consulta_3")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta_3:
                asistente(consulta_usuario_3)
    
    mostrar_botones_navegacion()
                    
elif st.session_state.subpagina == "Generación automática de reportes académicos":
    st.write("# Generación automática de reportes académicos: ")
    st.write("## Casos de uso: ")
    st.write("### **1-. Alertas automáticas sobre el rendimiento académico:**")
    with st.container():
        # Barra de texto para la consulta y botón de envío
        consulta_usuario_4 = st.text_area("Ejemplo de consulta:", 
                                        value="Lista a los estudiantes que hayan reprobado una o más asignaturas en el periodo (202303). Escribe un correo electrónico invitándoles a reunirse con algún profesor de la academia de matemáticas para recibir una tutoría académica. Muéstrame el correo para aprobarlo antes de que lo envíes.")
        enviar_consulta_4 = st.button("Enviar", key="enviar_consulta_3")

        # Si el usuario hace clic en "Enviar"
        with st.container(height=310):
            if enviar_consulta_4:
                asistente(consulta_usuario_4)
    
    mostrar_botones_navegacion()
                  
        
