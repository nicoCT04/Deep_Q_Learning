# Proyecto 2 — Deep Q-Learning en Space Invaders

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Nicolás Concuá**

Entrenamiento de un agente de Aprendizaje por Refuerzo (DQN, con
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)) capaz de jugar
**Space Invaders** sobre el entorno `ALE/SpaceInvaders-v5` de Gymnasium. El
proyecto documenta el proceso completo de desarrollo: análisis del entorno,
metodología, iteraciones de entrenamiento, evaluación y video del agente.

> El entrenamiento se realiza en **Google Colab (GPU)** con checkpoints
> periódicos a Google Drive como respaldo. El código local sirve para prototipar,
> analizar el entorno y evaluar/gravar el agente final.

## Estructura

```
Deep_Q_Learning/
├── notebook/
│   ├── proyecto2.ipynb   # investigación + entrenamiento (Colab) + evaluación + video
│   └── ale_utils.py      # utilidades reutilizadas del Lab5 + agente desde modelo
├── models/               # pesos del modelo final (.zip de SB3)
├── reports/              # figuras (.png) e informe (.docx/.pdf, no versionado)
└── videos/               # .mp4 del agente jugando
```

## Cómo reproducir

### Local (análisis del entorno, evaluación y video)

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install "gymnasium[atari,other]" ale-py stable-baselines3 numpy matplotlib opencv-python
cd notebook
jupyter nbconvert --to notebook --execute --inplace proyecto2.ipynb
```

Requiere `ffmpeg` en el sistema para escribir los videos `.mp4`.

### Entrenamiento (Google Colab)

Ver la sección de entrenamiento dentro de `notebook/proyecto2.ipynb`: monta Google
Drive, instala dependencias, entrena el DQN con GPU y guarda checkpoints en Drive
para poder reanudar si la sesión se corta.

## Cargar los pesos del modelo final

```python
from stable_baselines3 import DQN
modelo = DQN.load("models/dqn_spaceinvaders.zip")
```

El preprocesamiento del entorno de evaluación debe ser **idéntico** al de
entrenamiento (mismos wrappers, tamaño de frame y apilado de frames); el notebook
lo construye con la función `crear_entorno_atari`.
