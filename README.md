

uv init chatbot-phi3

uv add torch transformers datasets evaluate accelerate jupyter matplotlib numpy pandas
uv add tensorboard optimum onnxruntime

uv add --dev ipykernel                
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=chatbot-phi3
uv add --dev uv 


