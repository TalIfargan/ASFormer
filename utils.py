import os

def get_train_val_lists(split_num, fold_path, features_path):
    files = os.listdir(fold_path)
    train_list = []
    val_list = []
    test_list = []
    for file in files:
        file_path = os.path.join(fold_path, file)
        with open(file_path) as f:
            lines = f.read().splitlines()
            lines = [line[:line.index('.')] for line in lines]
        if file.startswith(f'valid {split_num}'):
            val_list += lines
        elif file.startswith(f'test {split_num}'):
            test_list += lines
    files_in_fold = os.listdir(features_path)
    files_in_fold = [file[:file.index('.')] for file in files_in_fold]
    train_list = [video for video in files_in_fold if video not in val_list + test_list]
    return train_list, val_list, test_list
