#set GPU
python3 backend_api.py --port 6006 --timeout 30000 --debug --model=incoder --gpu=2
python3 backend_api.py --port 6007 --timeout 30000 --debug --model=polycoder --gpu=2
python3 backend_api.py --port 6008 --timeout 30000 --debug --model=codellama --gpu=3
python3 backend_api.py --port 6009 --timeout 30000 --debug --model=starcoder2 --gpu=2