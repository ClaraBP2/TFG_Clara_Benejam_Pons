# TFG_Clara_Benejam_Pons
Sistema inteligente distribuido basado en Edge Computing y Aprendizaje  Federado para la monitorización del comportamiento animal 

## Requisitos

### Entorno de desarrollo

* Python 3.10
* Sistema operativo Windows o Linux
* NVIDIA Jetson Nano (para las pruebas en Edge Computing)
* Conexión USB o red local entre servidor y cliente. En este caso se detalla la opción de USB. 

### Instalación de dependencias

Se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
python -m venv venv
```

Activar el entorno virtual:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Jetson Nano**

```bash
source venv/bin/activate
```

Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

### Principales librerías utilizadas


| Librería            | Propósito                             |
| ------------------- | ------------------------------------- |
| TensorFlow 2.13.1   | Entrenamiento y evaluación de modelos |
| Flower 0.19.0       | Aprendizaje federado                  |
| Scikit-learn 1.8.0  | Preprocesado y métricas               |
| Pandas 3.0.1        | Manipulación de datos                 |
| NumPy 1.26.4        | Operaciones numéricas                 |
| Matplotlib 3.10.8   | Visualización de resultados           |
| SQLAlchemy 2.0.49   | Gestión de almacenamiento y registros |
| TensorBoard 2.13.0  | Monitorización del entrenamiento      |
| Cryptography 41.0.7 | Funciones criptográficas y seguridad  |

### Archivo requirements.txt

Las versiones exactas de todas las dependencias utilizadas durante el desarrollo se encuentran especificadas en el archivo:

```text
requirements.txt
```

Para garantizar la reproducibilidad de los experimentos se recomienda utilizar las versiones indicadas en dicho archivo. Se recomineda encarecidamente ya que es complexo encontrar librerías compatibles. 
