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
> analizar el entorno y evaluar/grabar el agente final.

## Resultados

Agente **DQN** (`CnnPolicy`) entrenado durante **8 000 000 de pasos** (localmente
en un MacBook Pro M4 Pro con MPS). Evaluación con política greedy (ε = 0), episodio
completo de 3 vidas y puntaje real (sin *reward clipping*), sobre 20 episodios:

| Métrica | Valor |
|:--------|:------|
| **Puntaje máximo (métrica de competencia)** | **1815** |
| Puntaje promedio | 1022 |
| Modelo inicial (2 M pasos, Colab) | 1255 / 625 |
| Baseline aleatorio (referencia) | ~150 |

Se compararon tres iteraciones (2 M base, 8 M base y 8 M con LR 2.5e-4); la de
**8 M con LR base** fue la mejor. Videos del agente jugando:
`videos/agente_dqn_final.mp4` (partida completa) y `videos/agente_dqn_1880.mp4`
(clip de la mejor partida, 1880 puntos).

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

Abrir `notebook/proyecto2.ipynb` en Colab, activar **GPU (T4)** y correr la
sección 2: monta Google Drive, instala dependencias, entrena el DQN con GPU y
guarda checkpoints cada 100 k pasos en `MyDrive/Proyecto2_SpaceInvaders/`. Si la
sesión se corta, al re-ejecutar la celda 2.10 el entrenamiento **reanuda solo**
desde el último checkpoint.

## Cargar los pesos y evaluar el agente

```python
import gymnasium as gym, ale_py
gym.register_envs(ale_py)
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# 1) Cargar el modelo entrenado
modelo = DQN.load("models/dqn_spaceinvaders.zip")

# 2) Entorno de evaluación con el MISMO preprocesamiento (puntaje real, 3 vidas)
env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1,
                     wrapper_kwargs=dict(clip_reward=False, terminal_on_life_loss=False))
env = VecFrameStack(env, n_stack=4)

# 3) Jugar un episodio greedy
obs = env.reset(); done = [False]; total = 0.0
while not done[0]:
    accion, _ = modelo.predict(obs, deterministic=True)
    obs, r, done, _ = env.step(accion); total += float(r[0])
print("Puntaje:", total)
```

El preprocesamiento del entorno de evaluación debe ser **idéntico** al de
entrenamiento (gris 84×84, apilado de 4 frames); solo se desactivan el *reward
clipping* y el *episodic-life* para medir el puntaje real de la competencia. El
notebook automatiza esto en la sección 4 (`crear_entorno_eval`), que también
genera el video.
