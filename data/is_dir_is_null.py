import os
from glob import glob
from tqdm import tqdm

# frames_pathes = sorted(glob(os.path.join("/datasets/CH-SIMS", "Processed/video/AlignedFaces", "*/*")))
#
# for frame in tqdm(frames_pathes):
#     if not os.listdir(frame):
#         print(frame)

csv_pathes = sorted(glob(os.path.join("/datasets/CH-SIMS", "Processed/video/OpenFace2", "*/*")))
for frame in tqdm(csv_pathes):
    # if not bool(glob(os.path.join(frame+'/*.csv'))):
    #     print(frame)

    if not os.listdir(frame):
        print(frame)