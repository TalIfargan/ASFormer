import os



def modify_txt(txt_path):
    file_ptr = open(txt_path, 'r')
    gt_file_content = file_ptr.read().split('\n')[:-1]
    new_gt_content = []
    for item in gt_file_content:
        start, end, gesture = int(item.split()[0]), int(item.split()[1]), item.split()[2]
        if end - start < 22:
            new_gt_content.append(f'{start} {end} {gesture}_start')
        else:
            new_gt_content.append(f'{start} {start+9} {gesture}_start')
            new_gt_content.append(f'{start+10} {end-10} {gesture}_middle')
            new_gt_content.append(f'{end-9} {end} {gesture}_end')
    return new_gt_content




for file in os.listdir('data/transcriptions_gestures/'):
    modified = modify_txt(os.path.join('data/transcriptions_gestures/', file))
    with open(f'data/gestures_modified/{file}', 'w') as f:
        for line in modified:
            f.write(line)
            f.write('\n')
