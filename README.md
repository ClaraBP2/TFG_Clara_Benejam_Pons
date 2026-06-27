# TFG_Clara_Benejam_Pons

# Sistema inteligente distribuido basado en Edge Computing y Aprendizaje Federado para la monitorización del comportamiento animal

Este proyecto implementa un sistema de **Aprendizaje Federado (Federated Learning)** utilizando **Flower** para el entrenamiento distribuido de modelos de clasificación del comportamiento animal. El sistema permite entrenar un modelo global a partir de datos distribuidos entre distintos clientes, incluyendo dispositivos **Edge** como la **NVIDIA Jetson Nano**.

---

# Requisitos

## Entorno de desarrollo

- Python 3.10
- Windows o Linux
- NVIDIA Jetson Nano (para las pruebas de ejecución en el Edge)
- Conexión USB o red local entre el servidor (ordenador personal) y el cliente (Jetson Nano). En este caso se describe la configuración mediante conexión USB.

## Instalación de dependencias

Se recomienda crear un entorno virtual antes de instalar las dependencias.

### Crear el entorno virtual

```bash
python -m venv venv
```

### Activar el entorno virtual

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Jetson Nano**

```bash
source venv/bin/activate
```

### Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Principales librerías utilizadas

| Librería | Descripción |
|----------|-------------|
| TensorFlow 2.13.1 | Entrenamiento y evaluación del modelo |
| Flower 0.19.0 | Framework de Aprendizaje Federado |
| Scikit-learn 1.8.0 | Preprocesado y métricas |
| Pandas 3.0.1 | Manipulación de datos |
| NumPy 1.26.4 | Operaciones numéricas |
| Matplotlib 3.10.8 | Visualización de resultados |

## Archivo `requirements.txt`

Todas las dependencias utilizadas durante el desarrollo se encuentran especificadas en el archivo:

```text
requirements.txt
```

Para garantizar la reproducibilidad de los experimentos, se recomienda utilizar exactamente las versiones indicadas, ya que algunas librerías presentan restricciones de compatibilidad entre versiones.

---

# Estructura del proyecto

```text
TFG_Clara_Benejam_Pons/
│
├── data_processing/
│   ├── clean_data.py         
│   └── dataset_analysis.py  
│
├── federated_learning/
│   ├── run_flower.py
│   ├── client_flower.py
│   ├── server_flower.py
│   ├── fine_tuning_flower.py
│   ├── data_loader.py
│   ├── evaluation.py
│   └── utils.py
│
├── model/
│   ├── classification_model.py
│   └── __init__.py
│
├── baselines/
│   └── centralizado.py
│
├── requirements.txt
└── README.md
```

---

# Preprocesado de los datos

Antes de iniciar el entrenamiento es necesario realizar el preprocesado del conjunto de datos.

El script `clean_data.py` realiza automáticamente:

- Conversión de variables a formato numérico.
- Recodificación de las etiquetas originales en tres clases.
- Generación de los conjuntos de datos limpios utilizados durante el entrenamiento federado.

Ejecutar:

```bash
python data_processing/clean_data.py
```

Los conjuntos de datos procesados se almacenan automáticamente en:

```text
data/clean/
```

manteniendo la organización original de los distintos rebaños.

---

# Configuración de la comunicación entre el servidor y la Jetson Nano

Para validar el funcionamiento del sistema en un entorno distribuido real se estableció una comunicación directa entre el ordenador personal, encargado de ejecutar el servidor Flower, y una **NVIDIA Jetson Nano**, utilizada como cliente Edge.

Ambos dispositivos se conectaron mediante una interfaz USB y se configuraron direcciones IP estáticas dentro de la misma subred.

## Configuración de red

| Dispositivo | Dirección IP |
|-------------|--------------|
| Ordenador personal (Servidor) | `192.168.55.100` |
| NVIDIA Jetson Nano (Cliente) | `192.168.55.1` |
| Puerto Flower | `8081` |

## Configuración de la dirección IP en Windows

Se necesita tener los Abrir PowerShell como administrador y ejecutar:

```powershell
New-NetIPAddress 
    -InterfaceAlias "Ethernet 3" 
    -IPAddress 192.168.55.100 
    -PrefixLength 24
```

> **Nota:** El nombre de la interfaz (`Ethernet 3`) puede variar según la configuración del ordenador. Tras conectar la Jetson Nano mediante el cable USB, diríjase al apartado **Red e Internet** de Windows y compruebe cuál es la interfaz de red que se ha habilitado. Sustituya `Ethernet 3` por el nombre correspondiente en el comando anterior. Acostumbra a ser Etehrnet 3 o 6. 


## Configuración temporal del Firewall

Durante las pruebas iniciales puede ser necesario deshabilitar temporalmente el Firewall de Windows.

Deshabilitar:

```powershell
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
```

Volver a habilitar:

```powershell
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
```

## Verificación de la conectividad

Para verificar la conectividad entre ambos dispositivos, ejecute ping 192.168.55.1 desde el ordenador o ping 192.168.55.100 desde la Jetson Nano.

---

# Ejecución del entrenamiento federado

## 1. Preprocesar los datos

```bash
python data_processing/clean_data.py
```

## 2. Iniciar el servidor Flower

Desde el ordenador personal ejecutar:

```bash
python federated_learning/run_flower.py
```

Este script:

- Inicia el servidor Flower.
- Lanza automáticamente cuatro clientes simulados.
- Coordina las rondas de entrenamiento federado.

## 3. Iniciar el cliente remoto en la Jetson Nano

Desde la Jetson Nano ejecutar:

```bash
python3 client_flower.py \
    --client_id MureskDryPasture \
    --server_ip 192.168.55.100 \
    --server_port 8081
```

### Parámetros

| Parámetro | Descripción |
|-----------|-------------|
| `client_id` | Identificador único del cliente federado. |
| `server_ip` | Dirección IP del ordenador que ejecuta el servidor Flower. |
| `server_port` | Puerto utilizado por el servidor Flower. |

> **Importante:** El valor del parámetro `server_ip` debe adaptarse a la configuración de red utilizada en cada despliegue.

---

# Flujo de ejecución

1. Crear y activar el entorno virtual.
2. Instalar las dependencias.
3. Ejecutar el preprocesado de los datos.
4. Configurar la comunicación entre el servidor y la Jetson Nano.
5. Comprobar la conectividad mediante `ping`.
6. Iniciar el servidor Flower.
7. Ejecutar el cliente en la Jetson Nano.
8. Supervisar el entrenamiento desde la consola del servidor.

En caso de entrenar el modelo, se recomienda desactivar el modo de suspensión del ordenador para evitar que el proceso de entrenamiento se interrumpa.

---

# Hardware utilizado

## Servidor

- Ordenador personal
- Windows 11
- Python 3.10

## Cliente Edge

- NVIDIA Jetson Nano Developer Kit
- Ubuntu
- Python 3.10
- TensorFlow 2.13.1

---

# Autor

**Clara Benejam Pons**

Trabajo Fin de Grado

**Sistema inteligente distribuido basado en Edge Computing y Aprendizaje Federado para la monitorización del comportamiento animal.**
