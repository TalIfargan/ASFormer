import os


for i in range(5):
    os.system(f"python main.py --split {str(i)}")
    os.system(f"python main.py --split {str(i)} --action predict")
