
import subprocess
import threading


# Using 127.0.0.1 because localhost does not work properly in Colab

def run_controller():
    subprocess.run(["python3", "-m", "fastchat.serve.controller", "--host", "127.0.0.1"])

def run_model_worker():
    subprocess.run(["python3", "-m", "fastchat.serve.model_worker", "--host", "127.0.0.1", "--controller-address", "http://127.0.0.1:21001", "--model-path", "google/gemma-3-4b-it"])

def run_api_server():
    subprocess.run(["python3", "-m", "fastchat.serve.openai_api_server", "--host", "localhost", "--controller-address", "http://127.0.0.1:21001", "--port", "8000"])



# Start controller thread
# see `controller.log` on the local storage provided by Colab
controller_thread = threading.Thread(target=run_controller)
controller_thread.start()
     

# Start model worker thread

# see `controller.log` on the local storage provided by Colab
# important to wait until the checkpoint shards are fully downloaded
model_worker_thread = threading.Thread(target=run_model_worker)
model_worker_thread.start()

     

# Start API server thread
api_server_thread = threading.Thread(target=run_api_server)
api_server_thread.start()
     
