import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
import textworld
import os
import json
# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)


# interact
obs, info = env.reset()
dones = (False,)

gamefile = info["extra.gamefile"][0]       # e.g., /.../something.z8
basedir = os.path.dirname(gamefile)

# traj_data.json lives in the same folder
traj_file = os.path.join(basedir, "traj_data.json")

with open(traj_file, "r") as f:
    traj_data = json.load(f)

print("Task description:", traj_data["turk_annotations"]["anns"][0]["task_desc"])
#print("Expert plan:", traj_data["expert_plan"])

while not dones[0]:
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    random_actions = [np.random.choice(admissible_commands[0])]

    # step
    obs, scores, dones, infos = env.step(random_actions)
    #print(type(dones))
    print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))