"""The reference for plan 27's classic environments (phase R6): what Gymnasium's CartPole-v1, Pendulum-v1 and
MountainCar-v0 do from a given start state under a given sequence of actions, for Idrak.Reinforcement.Plugins.Classic to
be held to.

    python3 -m venv rl-venv && rl-venv/bin/pip install -r tools/reinforcement/requirements.txt
    rl-venv/bin/python -I tools/reinforcement/make_fixtures.py        # writes tests/Idrak.Tests/data/reinforcement/

Gymnasium draws its start states from numpy's generator, which Idrak does not imitate (it draws its own from the episode's
seed). So every episode here starts from a state this script draws (a fixed seed) and sets on the environment after its
reset (env.unwrapped.state), and Idrak's test starts its environment from the same state (ResetTo); the actions are drawn
here or chosen here by a fixed rule from Gymnasium's observation, recorded, and replayed by the test as they are. Each
environment is made with gymnasium.make (so the TimeLimit wrapper truncates at the registered step limit, or at the
episode's own max_episode_steps, which the test passes as the kind's "max-steps" option).

Files written (each holds the versions that ran and its episodes):

- cartpole.json, pendulum.json, mountain-car.json: per episode its name, the Idrak kind's options, Gymnasium's make
  arguments, the start state (float64), the actions, and what Gymnasium gave back: the observations (the first one the
  start state's, then one per step, as float32 values), the rewards and the terminated and truncated flags of each step.
- manifest.json: the library versions and each file's size and SHA-256.

An episode is kept only when it is robust to rounding: replayed from its start state changed by a few parts in 1e12 (the
size of a last-bit difference in a sine on another platform, grown by the dynamics), its observations move by less than
1e-6 (relative, at least 1 absolute) and its rewards and flags stay the same; otherwise the script stops naming it, so
the fixtures never hold an episode whose comparison would depend on the last bit of a platform's libm. The C# test then
compares within 1e-5 (plans/done/27-reinforcement.md, "R6 as built").
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "reinforcement")
SEED = 20261022
ROBUST = 1e-6          # the largest relative change of an observation a perturbed start may make
PERTURBATION = 4e-12   # the relative change of each start element (an absolute 1e-14 where it is 0)


def observe(kind, env):
    """The observation of the state set on `env` (the unwrapped environment), as the environment makes it."""
    if kind == "pendulum":
        return env._get_obs()
    return np.array(env.state, dtype=np.float32)


def set_state(kind, env, start):
    if kind == "pendulum":
        env.state = np.array(start, dtype=np.float64)
        env.last_u = None
    elif kind == "cartpole":
        env.state = np.array(start, dtype=np.float64)
        env.steps_beyond_terminated = None
    else:
        env.state = np.array(start, dtype=np.float64)


def gym_action(kind, action):
    if kind == "pendulum":
        return np.array([action], dtype=np.float32)
    return int(action)


def play(spec, start, actions=None, rule=None):
    """Plays one episode from `start`: the recorded `actions`, or those `rule` chooses from each observation."""
    kind = spec["kind"]
    env = gym.make(spec["id"], max_episode_steps=spec["max_episode_steps"], **spec["make"])
    env.reset(seed=0)
    set_state(kind, env.unwrapped, start)
    observation = observe(kind, env.unwrapped)
    played = {"actions": [], "observations": [observation], "rewards": [], "terminated": [], "truncated": []}
    step = 0
    while True:
        if actions is not None:
            if step == len(actions):
                break
            action = actions[step]
        else:
            action = rule(observation, step)
        observation, reward, terminated, truncated, _ = env.step(gym_action(kind, action))
        played["actions"].append(action)
        played["observations"].append(observation)
        played["rewards"].append(float(reward))
        played["terminated"].append(bool(terminated))
        played["truncated"].append(bool(truncated))
        step += 1
        if terminated or truncated:
            break
    env.close()
    return played


def perturbed(start):
    return [v * (1 + PERTURBATION * (1 if i % 2 == 0 else -1)) if v != 0 else 1e-14 for i, v in enumerate(start)]


def check_robust(name, spec, start, played):
    again = play(spec, perturbed(start), actions=played["actions"])
    if len(again["rewards"]) != len(played["rewards"]) or again["terminated"] != played["terminated"] or again["truncated"] != played["truncated"]:
        sys.exit(f"{name}: a start changed by {PERTURBATION} ends the episode otherwise ({len(again['rewards'])} steps, {len(played['rewards'])} played); choose another")
    worst = 0.0
    for a, b in zip(played["observations"], again["observations"]):
        for x, y in zip(a, b):
            worst = max(worst, abs(float(x) - float(y)) / max(1.0, abs(float(x))))
    rewards = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(played["rewards"], again["rewards"]))
    if worst > ROBUST or rewards > ROBUST:
        sys.exit(f"{name}: a start changed by {PERTURBATION} moves the observations by {worst:.3g} and the rewards by {rewards:.3g} (above {ROBUST}); choose another")
    return worst


def episode(name, spec, options, start, actions=None, rule=None):
    played = play(spec, start, actions=actions, rule=rule)
    worst = check_robust(name, spec, start, played)
    print(f"  {name}: {len(played['rewards'])} steps, terminated {played['terminated'][-1]}, truncated {played['truncated'][-1]}, "
          f"return {sum(played['rewards']):.4f}, perturbed start moved the observations by {worst:.2g}")
    return {
        "name": name,
        "options": options,
        "make": {k: v for k, v in spec["make"].items()},
        "max_episode_steps": spec["max_episode_steps"],
        "start": [float(v) for v in start],
        "actions": [float(a) if spec["kind"] == "pendulum" else int(a) for a in played["actions"]],
        "observations": [[float(v) for v in o] for o in played["observations"]],
        "rewards": played["rewards"],
        "terminated": played["terminated"],
        "truncated": played["truncated"],
    }


def spec(kind, env_id, max_steps, **make):
    return {"kind": kind, "id": env_id, "max_episode_steps": max_steps, "make": make}


def cartpole(rng):
    episodes = []
    default = spec("cartpole", "CartPole-v1", 500)
    for i in range(8):
        start = rng.uniform(-0.05, 0.05, size=4).tolist()
        actions = rng.integers(0, 2, size=500).tolist()
        episodes.append(episode(f"random actions {i + 1}", default, {}, start, rule=lambda o, s, a=actions: a[s]))
    for i in range(2):
        start = rng.uniform(-0.2, 0.2, size=4).tolist()
        actions = rng.integers(0, 2, size=500).tolist()
        episodes.append(episode(f"a wider start {i + 1}", default, {}, start, rule=lambda o, s, a=actions: a[s]))

    def balance(o, s):  # pushes the cart under the pole: keeps it up for a while
        return 1 if o[2] + 0.3 * o[3] + 0.01 * o[0] + 0.05 * o[1] > 0 else 0

    episodes.append(episode("balanced to a 60-step limit (truncated)", spec("cartpole", "CartPole-v1", 60), {"max-steps": "60"},
                            rng.uniform(-0.02, 0.02, size=4).tolist(), rule=balance))
    episodes.append(episode("out past x = 2.4 (terminated by position)", default, {}, [2.3, 1.0, 0.0, 0.0], rule=balance))
    actions = rng.integers(0, 2, size=500).tolist()
    episodes.append(episode("Sutton and Barto's reward", spec("cartpole", "CartPole-v1", 500, sutton_barto_reward=True), {"sutton-barto": "true"},
                            rng.uniform(-0.05, 0.05, size=4).tolist(), rule=lambda o, s, a=actions: a[s]))
    return episodes


def pendulum(rng):
    episodes = []
    default = spec("pendulum", "Pendulum-v1", 200)
    for i in range(3):
        start = [float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-1, 1))]
        torques = rng.uniform(-2.5, 2.5, size=200).astype(np.float32).tolist()      # past the limit of 2: clipped
        episodes.append(episode(f"random torques {i + 1} (to the 200-step limit)", default, {}, start, rule=lambda o, s, a=torques: a[s]))
    torques = rng.uniform(-2.0, 2.0, size=200).astype(np.float32).tolist()
    episodes.append(episode("another gravity (g = 9.81)", spec("pendulum", "Pendulum-v1", 200, g=9.81), {"g": "9.81"},
                            [float(rng.uniform(-math.pi, math.pi)), 0.5], rule=lambda o, s, a=torques: a[s]))
    episodes.append(episode("spun past the top speed of 8 (clipped)", spec("pendulum", "Pendulum-v1", 40), {"max-steps": "40"},
                            [0.1, 7.9], rule=lambda o, s: np.float32(2.0) if o[2] > 0 else np.float32(-2.0)))
    return episodes


def mountain_car(rng):
    episodes = []
    default = spec("mountain-car", "MountainCar-v0", 200)
    for i in range(3):
        actions = rng.integers(0, 3, size=200).tolist()
        episodes.append(episode(f"random actions {i + 1} (to the 200-step limit)", default, {}, [float(rng.uniform(-0.6, -0.4)), 0.0],
                                rule=lambda o, s, a=actions: a[s]))

    def pump(o, s):  # pushes along the velocity: builds up the swing to the goal
        return 2 if o[1] >= 0 else 0

    for i in range(2):
        episodes.append(episode(f"pumped to the goal {i + 1} (terminated)", default, {}, [float(rng.uniform(-0.6, -0.4)), 0.0], rule=pump))
    episodes.append(episode("into the left wall (velocity zeroed)", spec("mountain-car", "MountainCar-v0", 40), {"max-steps": "40"},
                            [-1.15, -0.05], rule=lambda o, s: 0))
    episodes.append(episode("a goal velocity of 0.01", spec("mountain-car", "MountainCar-v0", 200, goal_velocity=0.01), {"goal-velocity": "0.01"},
                            [float(rng.uniform(-0.6, -0.4)), 0.0], rule=pump))
    return episodes


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)
    versions = {"gymnasium": gym.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}
    files = {}
    for kind, env_id, make in (("cartpole", "CartPole-v1", cartpole), ("pendulum", "Pendulum-v1", pendulum), ("mountain-car", "MountainCar-v0", mountain_car)):
        print(f"{env_id}:")
        document = {"kind": kind, "environment": env_id, **versions, "episodes": make(rng)}
        path = os.path.join(OUT, f"{kind}.json")
        text = json.dumps(document, indent=1) + "\n"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        data = text.encode("utf-8")
        files[f"{kind}.json"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {**versions, "seed": SEED, "robust": ROBUST, "perturbation": PERTURBATION, "files": files}
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
