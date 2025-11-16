# How to use
```
pip install -r requirements.txt
cd .\hack_best\
uvicorn backend.main:app --reload --port 8000
```
# TODO create dockerfile!
```
sudo docker build -t best_hack_image .
sudo docker run --rm --name bh -p 8000:8000 best_hack_image
```
