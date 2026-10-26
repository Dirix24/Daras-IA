# -*- coding: utf-8 -*-
"""
Configuración central de Cerebro.

Todo lo que quieras "girar" está aquí: tamaño del modelo (perfiles),
hiperparámetros de entrenamiento, límites del agente y rutas. Ningún otro
archivo tiene números mágicos.

Puedes cambiar de perfil sin editar nada con una variable de entorno:

    CEREBRO_PERFIL=diminuto python3 train.py
    CEREBRO_PERFIL=mediano  python3 train.py
"""

import os

_RAIZ = os.path.abspath(os.path.dirname(__file__))

# ----------------------------------------------------------------------------
# PERFILES DE MODELO
# ----------------------------------------------------------------------------
# vocab_objetivo : tamaño de vocabulario que se intenta alcanzar con BPE
#                  (si el corpus es pequeño, BPE se detiene antes).
# block_size     : ventana de contexto máxima (tokens).
# dim / n_heads  : ancho del modelo y número de cabezas de atención.
# n_layers       : bloques transformer apilados.
# dim_ffn        : ancho de la capa feed-forward (típicamente 4 * dim).
# dropout        : regularización durante el entrenamiento (0 = apagado).
#
# "pequeno" es el perfil por defecto: ~0.2 M parámetros, pensado para un
# portátil viejo (la idea nació en un Asus X550LC): entrena en minutos.
PERFILES = {
    "diminuto": dict(vocab_objetivo=384,  block_size=96,  dim=32,  n_heads=2,
                     n_layers=2, dim_ffn=128, dropout=0.10),
    "pequeno":  dict(vocab_objetivo=512,  block_size=112, dim=64,  n_heads=4,
                     n_layers=3, dim_ffn=256, dropout=0.05),
    "mediano":  dict(vocab_objetivo=1024, block_size=160, dim=128, n_heads=4,
                     n_layers=4, dim_ffn=512, dropout=0.15),
    # V2: más capacidad y contexto, manteniendo NumPy puro.
    "v2":      dict(vocab_objetivo=768, block_size=192, dim=96, n_heads=4,
                     n_layers=4, dim_ffn=384, dropout=0.08),

}

PERFIL = os.environ.get("CEREBRO_PERFIL", "pequeno")
if PERFIL not in PERFILES:
    raise ValueError(
        f"Perfil desconocido '{PERFIL}'. Disponibles: {', '.join(PERFILES)}"
    )

PERFIL_ACTIVO = PERFILES[PERFIL]

# Tokenizador: "bpe" (recomendado, sin palabras desconocidas) o "palabras".
TOKENIZADOR = "bpe"

# Modelo
TIE_WEIGHTS = True       # comparte la matriz de embeddings con la capa de salida
DTYPE = "float32"        # "float32" (rápido) o "float64" (más preciso, más lento)

# ----------------------------------------------------------------------------
# ENTRENAMIENTO
# ----------------------------------------------------------------------------
SEED = 1234
BATCH_SIZE = 16
ACUMULACION_GRADIENTES = 2      # batch efectivo = BATCH_SIZE * este valor
TASA_APRENDIZAJE = 3e-3          # lr máximo (tras el calentamiento)
LR_MIN_FACTOR = 0.10             # el lr final es TASA_APRENDIZAJE * este factor
WARMUP_FRACCION = 0.05           # fracción de pasos con lr creciente
WEIGHT_DECAY = 0.05              # decaimiento de pesos desacoplado (AdamW)
GRAD_CLIP = 1.0                  # norma global máxima del gradiente
PASOS_POR_DEFECTO = 8000          # ~5-6 min en un portátil modesto con el perfil "pequeno"
EVAL_CADA = 200
PACIENCIA_EARLY_STOP = 8        # evaluaciones sin mejora antes de detenerse; 0 = desactivado
FRACCION_VALIDACION = 0.10       # proporción de documentos reservada a validación

# Optimizador: "adamw" (recomendado) o "sgd" (con momentum; más simple de razonar)
OPTIMIZADOR = "adamw"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
ADAM_EPS = 1e-8
SGD_MOMENTUM = 0.9

# ----------------------------------------------------------------------------
# GENERACIÓN (valores por defecto del chat)
# ----------------------------------------------------------------------------
GEN_MAX_NUEVOS = 60              # las respuestas del corpus miden hasta ~58 tokens
GEN_TEMPERATURA = 0.7
GEN_TOP_K = 40
GEN_TOP_P = 0.92
GEN_PENALIZACION_REPETICION = 1.15

# ----------------------------------------------------------------------------
# MEMORIA
# ----------------------------------------------------------------------------
MEMORIA_CORTO_PLAZO_TURNOS = 6   # turnos recientes que se incluyen en el prompt

# ----------------------------------------------------------------------------
# AGENTE Y LÍMITES DE SEGURIDAD
# ----------------------------------------------------------------------------
MAX_AGENT_STEPS = 4
MAX_TOOL_CALLS = 3
# Permisos que el agente tiene concedidos. Quita "WRITE" para un agente de
# solo lectura. Las herramientas exigen su permiso y se rechazan si falta.
PERMISOS_AGENTE = {"CALCULATE", "READ", "WRITE"}
MAX_FILE_BYTES = 1_000_000       # tope de lectura/escritura por archivo
WORKSPACE_DIR = os.path.join(_RAIZ, "data", "workspace")

# ----------------------------------------------------------------------------
# RUTAS
# ----------------------------------------------------------------------------
CORPUS_FILE = os.path.join(_RAIZ, "data", "corpus.txt")
CORPUS_DIR = os.path.join(_RAIZ, "data", "corpus")      # todos los .txt de aquí se añaden
VALIDACION_FILE = os.path.join(_RAIZ, "data", "validacion.txt")  # reformulaciones nuevas de cosas que sí sabe
GENERALIZACION_FILE = os.path.join(_RAIZ, "data", "generalizacion.txt")  # diálogos totalmente nuevos (nunca entrenados)
TEST_FILE = os.path.join(_RAIZ, "data", "test.txt")  # conjunto final: nunca selecciona checkpoints
MEMORY_FILE = os.path.join(_RAIZ, "memory", "memory_store.json")
DIR_CHECKPOINTS = os.path.join(_RAIZ, "checkpoints")
CKPT_MEJOR = os.path.join(DIR_CHECKPOINTS, "best.npz")   # el que usa el chat
CKPT_ULTIMO = os.path.join(DIR_CHECKPOINTS, "last.npz")  # con estado del optimizador
DIR_EVOLUTION = os.path.join(_RAIZ, "learning", "evolution")
