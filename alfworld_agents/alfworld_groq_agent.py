import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
import copy
import agent_load
from groq import Groq

MODEL_NAME = "gemma2-9b-it"
confidence_probe_prompt = "before you output the action for the next step, first output a set of confidence values that reflects the confidence you have in finding a ladle in each of the following locations [cabinet 1', 'cabinet 10', 'cabinet 11', 'cabinet 12', 'cabinet 13', 'cabinet 14', 'cabinet 15', 'cabinet 16', 'cabinet 17', 'cabinet 18', 'cabinet 19', 'cabinet 2', 'cabinet 20', 'cabinet 21', 'cabinet 22', 'cabinet 23', 'cabinet 24', 'cabinet 25', 'cabinet 26', 'cabinet 27', 'cabinet 3', 'cabinet 4', 'cabinet 5', 'cabinet 6', 'cabinet 7', 'cabinet 8', 'cabinet 9', 'coffeemachine 1', 'countertop 1', 'countertop 2', 'diningtable 1', 'drawer 1', 'drawer 10', 'drawer 11', 'drawer 12', 'drawer 2', 'drawer 3', 'drawer 4', 'drawer 5', 'drawer 6', 'drawer 7', 'drawer 8', 'drawer 9', 'fridge 1', 'garbagecan 1', 'microwave 1', 'sinkbasin 1']. A confidence value between 0 to 100 should be assigned to all locations given above, the total of all confidence values should sum to 100"

def probe_confidence(chat_history, client):

        #last_user_msg = {"role" : "user" , 
        #"content" : f"{confidence_probe_prompt}"}
        with open("probe_model.json", "r") as f:
            last_user_msg = json.load(f)

        chat_history_cp = copy.deepcopy(chat_history)
        chat_history_cp.extend(last_user_msg)
        #print(chat_history_cp)
        
        # Generate Gemma response
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_history_cp,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stream=False, # <<< CHANGED: Set streaming to False
            stop=None,
            tools=[]
        )

        response = completion.choices[0].message.content
        print(response)

def gemma_alfworld_step_generator(env,task_desc, base_obs,base_info, client, chat_json_path, expert_plan=None):
    """
    Generator function to step through ALFWorld with Gemma.
    
    Parameters:
        env: ALFWorld environment (already initialized)
        model: Hugging Face Gemma model
        tokenizer: corresponding tokenizer
        chat_json_path: path to JSON file containing chat history (list of dicts)
        expert_plan: optional expert action list for fallback
    
    Yields:
        obs: current observation
        admissible_commands: list of admissible commands
        response: Gemma's output
        chat_history: updated chat history
    """
    
    # Reset environment
    
  
 
    # Load chat history JSON
    with open(chat_json_path, "r") as f:
        chat_history = json.load(f)
    
    # Ensure system role exists with task description
    if not any(msg["role"] == "system" for msg in chat_history):
        chat_history.insert(0, {"role": "system",
                                 "content": f"You are an agent in ALFWorld. Task: {task_desc}. "
                                            "Always respond with exactly one valid action from the admissible commands."})
    dones = (False,)
                                
    step = 1
    while not dones[0]:
        
        
        if step == 1:
          current_obs =  f"{base_obs[0]} , (Hint: check diningtable first)"
          admissible_commands = base_info['admissible_commands'][0]        
        else:
            current_obs = obs[0]
            admissible_commands = info['admissible_commands'][0]
        step+=1
        
        # Use last user message from chat history if exists, otherwise auto-generate
        user_messages = [msg for msg in chat_history if msg["role"] == "user"]

        last_user_msg = {"role": "user",
                            "content": f"Task : {task_desc} Observation: {current_obs}\nValid actions: {admissible_commands}"}
        chat_history.append(last_user_msg)
        
        # Generate Gemma response
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_history,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False, # <<< CHANGED: Set streaming to False
            stop=None,
            tools=[]
        )

        response = completion.choices[0].message.content
        
        # Pick a valid action from response or fallback
        action = None
        for cmd in admissible_commands:
            if cmd in response:
                action = cmd
                break
        if action is None:
            if "ACTION" in response:
                description = response.split("ACTION", 1)[1].strip()
                print(description)
                action = description 
            else:
                action = response  # fallback safe action
        
        chat_history.append({"role": "assistant", "content": action})
        
        with open(chat_json_path, "w") as f:
            json.dump(chat_history, f, indent=2)
        
        # Step environment
        obs, scores, dones, info = env.step([action])
        
        # Yield current step info
        yield current_obs, admissible_commands, response, chat_history, dones

if __name__ == '__main__':
    config = generic.load_config()
    env_type = config['env']['type']
    env = get_environment(env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

    obs, info = env.reset()
    print(obs)
    dones = (False,)
    #print(info["extra.expert_plan"])

    gamefile = info["extra.gamefile"][0]       # e.g., /.../something.z8
    basedir = os.path.dirname(gamefile)

    # traj_data.json lives in the same folder
    traj_file = os.path.join(basedir, "traj_data.json")
    print(traj_file)

    with open(traj_file, "r") as f:
        traj_data = json.load(f)
    #print(traj_data.keys())

    task_description = obs[0].partition("Your task is to: ")[-1]
    print(task_description)
    client = Groq()

    # Path to your chat history JSON
    chat_json_path = "chat_history.json"

    # Create generator
    gen = gemma_alfworld_step_generator(env, task_description, obs, info, client, chat_json_path)

    while True:
        try:
            obs, admissible, response, chat_history, dones = next(gen)
            print("\nObservation:", obs)
            print("Admissible actions:", admissible)
            print("model response:", response)

            probe_yes = input("add confidence probe? (y/n)").strip().lower()
            if probe_yes == 'y':
                print(probe_confidence(chat_history,client))
            # Ask user if they want to step forward
            user_input = input("\nRun next step? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Stopping generator.")
                break

        except StopIteration:
            print("Generator finished.")
            break