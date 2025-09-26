import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

def gemma_alfworld_step_generator(env,task_desc, base_info, model, tokenizer, chat_json_path, expert_plan=None):
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
          current_obs =  f"You are an agent in ALFWorld. Task: {task_desc}. Always respond with exactly one valid action from the admissible actions." 
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
        prompt_text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
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
    dones = (False,)

    gamefile = info["extra.gamefile"][0]       # e.g., /.../something.z8
    basedir = os.path.dirname(gamefile)

    # traj_data.json lives in the same folder
    traj_file = os.path.join(basedir, "traj_data.json")

    with open(traj_file, "r") as f:
        traj_data = json.load(f)

    task_description = traj_data["turk_annotations"]["anns"][0]["task_desc"]
    print(task_description)

    model_name = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

    # Path to your chat history JSON
    chat_json_path = "chat_history.json"

    # Create generator
    gen = gemma_alfworld_step_generator(env, task_description, info, model, tokenizer, chat_json_path)

    while True:
        try:
            obs, admissible, response, chat_history, dones = next(gen)
            print("\nObservation:", obs)
            print("Admissible actions:", admissible)
            print("Gemma response:", response)

            # Ask user if they want to step forward
            user_input = input("\nRun next step? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Stopping generator.")
                break

        except StopIteration:
            print("Generator finished.")
            break