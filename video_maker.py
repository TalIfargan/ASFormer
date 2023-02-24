import cv2
import os
import numpy as np


action_map = {"G0": "No Gesture",
              "G1": "Needle Passing",
              "G2": "Pull the Suture",
              "G3": "Instrument Tie",
              "G4": "Lay the Knot" ,
              "G5": "Cut the Suture"}



font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
color = (255, 0, 0)
thickness = 2


def add_segmentation(img, i, base, v2, num_frames, gt):
    frame = cv2.imread(img)
    width = frame.shape[1]
    height = frame.shape[0]
    base = cv2.imread(base)
    v2 = cv2.imread(v2)
    left_size = 150

    patch_header = np.full((200, v2.shape[1], v2.shape[2]), 255, dtype=np.uint8)
    patch_header = cv2.putText(patch_header, f'GT: {gt}', (500, 80), font, fontScale, color, thickness, cv2.LINE_AA)
    header_left = np.full((patch_header.shape[0], left_size, patch_header.shape[2]), 255, dtype=np.uint8)
    frame_relative_part = int((i / num_frames) * patch_header.shape[1])
    patch_header[120:, frame_relative_part - 1:frame_relative_part + 1, :] = 0
    header_final = cv2.hconcat([header_left, patch_header])
    header_final = cv2.resize(header_final, (width, int((width / header_final.shape[1]) * header_final.shape[0])), interpolation=cv2.INTER_AREA)

    patch_1 = v2[:v2.shape[0] // 2, :, :]

    patch_1_left = np.full((patch_1.shape[0], left_size, patch_1.shape[2]), 255, dtype=np.uint8)
    patch_1_left = cv2.putText(patch_1_left, 'GT', (50, 65), font, 1, (32, 32, 32), thickness, cv2.LINE_AA)
    patch_1_left = cv2.putText(patch_1_left, 'V2', (50, 180), font, 1, (32, 32, 32), thickness, cv2.LINE_AA)
    patch_1_final = cv2.hconcat([patch_1_left, patch_1])
    patch_1_final = cv2.resize(patch_1_final, (width, int((width / patch_1_final.shape[1]) * patch_1_final.shape[0])), interpolation=cv2.INTER_AREA)

    patch_2 = base[base.shape[0] // 4:base.shape[0] // 2, :, :]
    patch_2_left = np.full((patch_2.shape[0], left_size, patch_2.shape[2]), 255, dtype=np.uint8)
    patch_2_left = cv2.putText(patch_2_left, 'Baseline', (10, 70), font, 1, (32, 32, 32), thickness, cv2.LINE_AA)
    patch_2_final = cv2.hconcat([patch_2_left, patch_2])
    patch_2_final = cv2.resize(patch_2_final, (width, int((width / patch_2_final.shape[1]) * patch_2_final.shape[0])), interpolation=cv2.INTER_AREA)

    image_with_patch = cv2.vconcat([frame, header_final, patch_1_final, patch_2_final])
    
    return image_with_patch

def image_seq_to_segmentation_video(images_path, baseline_path, v2_path, gts, output_path='./video.mp4', fps=30.0):
    img_array = []
    # loop over all frames
    base = cv2.imread(baseline_path)
    v2 = cv2.imread(v2_path)
    frames = sorted(os.listdir(images_path))
    num_frames = len(frames)
    for i, filename in enumerate(frames):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        image_with_segmentation = add_segmentation(img, i, base, v2, num_frames, gts[i])
        img_array.append(image_with_segmentation)

    print(size)
    print("writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("saved video @ ", output_path)

def extract_labels(labels_file):
    file_ptr = open(labels_file, 'r')
    gt_file_content = file_ptr.read().split('\n')[:-1]
    content = []
    for item in gt_file_content:
        splitted_item = item.split()
        content += [splitted_item[2]]*(int(splitted_item[1]) - int(splitted_item[0]) + 1)
    return [action_map[item] for item in content]

if __name__ == '__main__':
    gts = [os.path.join('data', 'transcriptions_gestures', vid+'.txt') for vid in ['P020_balloon1', 'P032_tissue1', 'P038_balloon2']]
    gts = [extract_labels(item) for item in gts]
    images_paths = ['/datashare/APAS/frames/P020_balloon1_side', '/datashare/APAS/frames/P032_tissue1_side', '/datashare/APAS/frames/P038_balloon2_side']
    baseline_segmentations = ['results/split_0/P020_balloon1_stage3.png', 'results/split_2/P032_tissue1_stage3.png', 'results/split_4/P038_balloon2_stage3.png']
    advanced_segmentations = ['results_v2_hidden_1280/split_0/P020_balloon1_stage3.png', 'results_v2_hidden_1280/split_2/P032_tissue1_stage3.png', 'results_v2_hidden_1280/split_4/P038_balloon2_stage3.png']
    for i in range(3):
        image_seq_to_segmentation_video(images_paths[i], baseline_segmentations[i], advanced_segmentations[i], gts[i], f'videos/{images_paths[i].split("/")[-1]}.mp4')