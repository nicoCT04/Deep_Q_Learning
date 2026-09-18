"""
ale_utils.py — Módulo de funciones reutilizables para interactuar con el
Arcade Learning Environment (ALE) a través de Gymnasium.

Proyecto 2 (Deep Q-Learning) — CC3092 Deep Learning y Sistemas Inteligentes (UVG)
Nicolás Concuá

Base de infraestructura reutilizada del Laboratorio #5: crea entornos de Atari,
ejecuta agentes y graba video de las partidas. Para el Proyecto 2 se añade la
sección 5, que envuelve un modelo entrenado (p. ej. un DQN de Stable-Baselines3)
como una función-agente greedy intercambiable con ``agente_aleatorio`` y
``agente_regla_simple`` dentro de ``ejecutar_episodio`` / ``generar_video_agente``.

Las funciones están escritas para ser genéricas: aunque el objetivo es
ALE/SpaceInvaders-v5, funcionan con cualquier entorno de Gymnasium.
"""

import os
import gymnasium as gym

# ale-py registra los entornos ALE/* en Gymnasium. A partir de gymnasium 1.0
# es necesario llamar a register_envs para exponerlos a gym.make.
import ale_py

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# 1. Creación de entornos (con grabación de video opcional)
# ---------------------------------------------------------------------------
def crear_entorno(
    nombre_entorno,
    video_folder=None,
    name_prefix="rl-video",
    episode_trigger=None,
    render_mode="rgb_array",
    **kwargs,
):
    """Crea y retorna un entorno de Gymnasium.

    Si se especifica ``video_folder``, envuelve el entorno con el wrapper
    ``gymnasium.wrappers.RecordVideo`` para grabar los episodios indicados por
    ``episode_trigger``. Para grabar video se necesita ``render_mode="rgb_array"``,
    que se fija por defecto.

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno, p. ej. ``"ALE/SpaceInvaders-v5"`` o ``"CartPole-v1"``.
    video_folder : str | None
        Carpeta donde escribir los .mp4. Si es None, no se graba video.
    name_prefix : str
        Prefijo de los archivos de video generados.
    episode_trigger : callable | None
        Función ``episodio(int) -> bool`` que decide qué episodios grabar.
        Por defecto graba todos los episodios.
    render_mode : str
        Modo de render; ``"rgb_array"`` es el requerido para grabar.
    **kwargs :
        Parámetros extra que se pasan a ``gym.make`` (p. ej. ``frameskip``,
        ``repeat_action_probability``, ``full_action_space``, ``obs_type``).

    Retorna
    -------
    gymnasium.Env
        El entorno, posiblemente envuelto en RecordVideo.
    """
    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)

    if video_folder is not None:
        os.makedirs(video_folder, exist_ok=True)
        if episode_trigger is None:
            # Por defecto: grabar todos los episodios.
            episode_trigger = lambda episodio: True
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=episode_trigger,
        )
    return env


# ---------------------------------------------------------------------------
# 2. Agentes de referencia (sin entrenamiento)
# ---------------------------------------------------------------------------
def agente_aleatorio(observation, env):
    """Agente baseline: retorna una acción muestreada del espacio de acción.

    Ignora la observación y devuelve ``env.action_space.sample()``. Sirve como
    línea base contra la cual comparar cualquier agente entrenado.
    """
    return env.action_space.sample()


def agente_regla_simple(observation, env):
    """Agente de regla simple (heurístico, no aprendido) para Space Invaders.

    Estrategia: disparar de forma agresiva mientras alterna el movimiento
    lateral, de modo que la nave patrulla la pantalla y dispara casi siempre.
    Usa las acciones combinadas RIGHTFIRE (4) y LEFTFIRE (5), que mueven y
    disparan a la vez, e intercala FIRE (1). No usa la observación (es una
    regla fija); la firma ``(observation, env)`` se mantiene para que sea
    intercambiable con ``agente_aleatorio`` en ``ejecutar_episodio``.

    Si el entorno no tiene al menos 6 acciones (no es Space Invaders con el
    action space por defecto), cae de vuelta a una acción aleatoria.
    """
    n = env.action_space.n if hasattr(env.action_space, "n") else 0
    if n < 6:
        # No es el espacio de acción de Space Invaders: usar baseline.
        return env.action_space.sample()

    # Contador interno de pasos guardado en la función para alternar el patrón.
    paso = getattr(agente_regla_simple, "_paso", 0)
    agente_regla_simple._paso = paso + 1

    # Space Invaders (action space por defecto):
    #   0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT, 4 RIGHTFIRE, 5 LEFTFIRE
    ciclo = paso % 12
    if ciclo < 5:
        return 4  # RIGHTFIRE: avanza a la derecha disparando
    elif ciclo == 5:
        return 1  # FIRE: dispara sin moverse
    elif ciclo < 11:
        return 5  # LEFTFIRE: avanza a la izquierda disparando
    else:
        return 1  # FIRE


# ---------------------------------------------------------------------------
# 3. Ejecución de un episodio completo
# ---------------------------------------------------------------------------
def ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None):
    """Ejecuta un episodio completo con ``funcion_agente``.

    Corre hasta que ``terminated`` o ``truncated`` sea verdadero, o hasta
    alcanzar ``max_steps``.

    Parámetros
    ----------
    env : gymnasium.Env
        Entorno ya creado (posiblemente con RecordVideo).
    funcion_agente : callable
        Función ``(observation, env) -> action``, p. ej. ``agente_aleatorio``.
    max_steps : int
        Tope de pasos para evitar episodios infinitos.
    seed : int | None
        Semilla para ``env.reset`` (reproducibilidad).

    Retorna
    -------
    dict
        ``{"pasos": int, "recompensa": float}`` con el número de pasos
        ejecutados y el return (recompensa total acumulada) del episodio.
    """
    obs, info = env.reset(seed=seed)
    pasos = 0
    recompensa_total = 0.0
    terminated = truncated = False

    while not (terminated or truncated) and pasos < max_steps:
        accion = funcion_agente(obs, env)
        obs, recompensa, terminated, truncated, info = env.step(accion)
        recompensa_total += float(recompensa)
        pasos += 1

    return {"pasos": pasos, "recompensa": recompensa_total}


# ---------------------------------------------------------------------------
# 4. Función de alto nivel: generar video de un agente
# ---------------------------------------------------------------------------
def generar_video_agente(
    nombre_entorno,
    funcion_agente,
    video_folder,
    name_prefix,
    n_episodios=1,
    max_steps=10000,
    seed=0,
    **kwargs,
):
    """Crea el entorno con grabación, ejecuta episodios y guarda los videos.

    Combina ``crear_entorno``, ``ejecutar_episodio`` y el cierre correcto del
    entorno (``env.close()``, indispensable para que RecordVideo vuelque el
    .mp4 a disco).

    Parámetros
    ----------
    nombre_entorno : str
        Id del entorno (p. ej. ``"ALE/SpaceInvaders-v5"``).
    funcion_agente : callable
        Agente a usar, p. ej. ``agente_aleatorio`` o ``agente_regla_simple``.
    video_folder : str
        Carpeta de salida de los videos.
    name_prefix : str
        Prefijo de los archivos .mp4.
    n_episodios : int
        Número de episodios completos a ejecutar y grabar.
    max_steps : int
        Tope de pasos por episodio.
    seed : int
        Semilla base; cada episodio usa ``seed + i`` para variar la partida.
    **kwargs :
        Parámetros extra para ``gym.make`` vía ``crear_entorno``.

    Retorna
    -------
    dict
        ``{"videos": [rutas .mp4], "metricas": [{"episodio", "pasos",
        "recompensa"}, ...]}``.
    """
    os.makedirs(video_folder, exist_ok=True)
    env = crear_entorno(
        nombre_entorno,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda episodio: True,  # grabar todos
        **kwargs,
    )

    metricas = []
    for i in range(n_episodios):
        resultado = ejecutar_episodio(
            env, funcion_agente, max_steps=max_steps, seed=seed + i
        )
        metricas.append(
            {
                "episodio": i,
                "pasos": resultado["pasos"],
                "recompensa": resultado["recompensa"],
            }
        )

    # Cerrar el entorno es indispensable para que los .mp4 se escriban a disco.
    env.close()

    # Recolectar las rutas de los videos generados con este prefijo.
    videos = sorted(
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.startswith(name_prefix) and f.endswith(".mp4")
    )

    return {"videos": videos, "metricas": metricas}


# ---------------------------------------------------------------------------
# 5. Agente a partir de un modelo entrenado (Proyecto 2)
# ---------------------------------------------------------------------------
def agente_desde_modelo(modelo, deterministic=True):
    """Convierte un modelo entrenado en una función-agente ``(obs, env) -> accion``.

    Devuelve una función compatible con ``ejecutar_episodio`` y
    ``generar_video_agente`` que consulta la política del modelo. Con
    ``deterministic=True`` usa la política greedy (sin exploración), que es la
    configuración exigida para la evaluación de la competencia.

    Está pensada para modelos de Stable-Baselines3 (``modelo.predict``), pero
    funciona con cualquier objeto que exponga un método ``predict(obs) -> (accion, estado)``.

    Importante: el entorno pasado a ``ejecutar_episodio`` debe aplicar el MISMO
    preprocesamiento (wrappers, tamaño de frame, apilado de frames) usado durante
    el entrenamiento; de lo contrario la observación no coincide con lo que la red
    espera. Ver ``crear_entorno_atari`` en el notebook del proyecto.

    Parámetros
    ----------
    modelo : objeto con ``predict``
        Modelo entrenado (p. ej. ``stable_baselines3.DQN``).
    deterministic : bool
        Si es True, política greedy (argmax de los valores Q). Recomendado para
        evaluación. Si es False, se permite la exploración interna del modelo.

    Retorna
    -------
    callable
        Función ``(observation, env) -> action``.
    """

    def _agente(observation, env):
        accion, _estado = modelo.predict(observation, deterministic=deterministic)
        # SB3 puede devolver la acción como arreglo (por el manejo vectorizado);
        # se normaliza a un entero para env.step en un entorno no vectorizado.
        try:
            return int(accion)
        except (TypeError, ValueError):
            return accion

    return _agente
