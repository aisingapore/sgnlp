bind = "0.0.0.0:8000"
wsgi_app = "api:app"
timeout = 180
workers = 4
preload_app = True
raw_env = ["PYTHON_PATH=../../", "TOKENIZERS_PARALLELISM=false"]
