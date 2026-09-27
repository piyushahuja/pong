import gymnasium as gym
import ale_py


# gym.registry.keys()          # all registered ids
gym.spec("ALE/Pong-v5")      # EnvSpec: entry_point, default kwargs, max steps
# gym.pprint_registry()


env = gym.make('ALE/Pong-v5', render_mode="human")

obs, info = env.reset()


print(obs.shape)
print(info)
print("Actions:", env.unwrapped.get_action_meanings())

print("Action space:", env.action_space.sample())
print(ale_py.Action)



for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs.shape, reward, terminated, truncated, info)
    if truncated or terminated:
        obs, info = env.reset()


env.close()


