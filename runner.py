import os


for i in range(5):
    os.system(f"python main.py --split {str(i)} --hidden 128")
    os.system(f"python main.py --split {str(i)} --action predict --hidden 128")

    os.system(f"python main.py --split {str(i)} --hidden 64")
    os.system(f"python main.py --split {str(i)} --action predict --hidden 64")

# for i in range(32):
#     os.system(f"python feature_maker.py")

# os.system(f"python pca_maker.py --split 0 --output_size 128")
# os.system(f"python pca_maker.py --split 1 --output_size 128")
# os.system(f"python pca_maker.py --split 2 --output_size 128")
# os.system(f"python pca_maker.py --split 3 --output_size 128")
# os.system(f"python pca_maker.py --split 4 --output_size 128")
# os.system(f"python pca_maker.py --split 0 --output_size 64")
# os.system(f"python pca_maker.py --split 1 --output_size 64")
# os.system(f"python pca_maker.py --split 2 --output_size 64")
# os.system(f"python pca_maker.py --split 3 --output_size 64")
# os.system(f"python pca_maker.py --split 4 --output_size 64")