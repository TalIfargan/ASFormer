import os


for i in range(1, 5):
    os.system(f"python main.py --split {str(i)}")
    os.system(f"python main.py --split {str(i)} --action predict")

# for i in range(32):
#     os.system(f"python feature_maker.py")
