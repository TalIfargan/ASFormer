import os


for i in range(3,5):
    os.system(f"python train_efficient.py --FOLD {str(i)}")