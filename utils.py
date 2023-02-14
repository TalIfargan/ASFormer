import os

def get_train_val_lists(split_num, fold_path):
    files = os.listdir(fold_path)
    train_list = []
    val_list = []
    test_list = []
    for file in files:
        file_path = os.path.join(fold_path, file)
        with open(file_path) as f:
            lines = f.read().splitlines()
        if file.startswith(f'valid {split_num}'):
            val_list += lines
        elif file.startswith('valid'):
            train_list += lines
        elif file.startswith(f'test {split_num}'):
            test_list += lines
    return train_list, val_list, test_list
