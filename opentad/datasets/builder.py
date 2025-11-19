import json
import torch
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset
from collections.abc import Sequence
from mmengine.registry import Registry, build_from_cfg, TRANSFORMS

DATASETS = Registry("dataset")
PIPELINES = TRANSFORMS

from collections import defaultdict
 
def sample_dataset_by_video(dataset, seed=42, max_samples=300):

    random.seed(seed)
 
    # 1. 创建 video_name -> indices & labels 映射

    video_to_indices = defaultdict(list)

    video_to_labels = defaultdict(set)

    video_label_table = [
        (item['metas']['video_name'], item['gt_labels'])
        for item in dataset
    ]
    print(f"video_label_table: {video_label_table[:10]}")  # 打印前10个视频名称和标签
    print(f"video_label_table length: {len(video_label_table)}")  # 打印视频数量
 
    for idx, (video_name, label) in enumerate(video_label_table):

        video_to_indices[video_name].append(idx)

        video_to_labels[video_name].update(int(l) for l in label)
 
    # 2. 打乱 video 顺序
    # print(f"video_to_labels: {video_to_labels[video_name]}")
    all_video_names = list(video_to_indices.keys())

    # random.shuffle(all_video_names)
 
    # 3. 选择覆盖所有标签（0~19）的最小 video 集合

    selected_indices = []
    selected_video_names = set()

    covered_labels = set()
 
    for video_name in all_video_names:

        video_labels = video_to_labels[video_name]

        if not video_labels.issubset(covered_labels):  # 至少有一个新标签

            selected_indices.extend(video_to_indices[video_name])
            selected_video_names.add(video_name)

            covered_labels.update(video_labels)

        if len(covered_labels) == 200:  # 已覆盖所有标签

            break
    
    if len(selected_video_names) < max_samples:
        remaining_videos = [v for v in all_video_names if v not in selected_video_names]
        random.shuffle(remaining_videos)
        for video_name in remaining_videos:
            if len(selected_video_names) >= max_samples:
                break
            selected_indices.extend(video_to_indices[video_name])
            selected_video_names.add(video_name)

    # 4. 创建子集
    print(f"selected_indices: {selected_indices}")


    dataset = CustomSubset(dataset, selected_indices)

    return dataset

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.__dict__.update(dataset.__dict__)

def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset, batch_size, rank, world_size, shuffle=False, drop_last=False, max_samples=None, seed=42, thumos=1, dataset_name=None, **kwargs):
    if dataset_name == 'charades':
        random.seed(seed)
        subset_indices = list(range(len(dataset)))
        random.Random(seed).shuffle(subset_indices)
        subset_indices = subset_indices[:100]
        print(f"subset_indices: {subset_indices}")
        dataset = CustomSubset(dataset, subset_indices)
    elif max_samples==10:
        # random.seed(seed)
        # subset_indices = list(range(len(dataset)))
        # random.shuffle(subset_indices)
        # subset_indices = subset_indices[:max_samples]        
        subset_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 25, 26, 27, 28, 29, 31, 32, 37, 38, 42, 43, 44, 45, 46, 47, 48, 74, 83, 84, 94, 95, 96, 97, 98, 124, 144, 145, 146]
        dataset = CustomSubset(dataset, subset_indices)
    elif max_samples==1:
        dataset = sample_dataset_by_video(dataset, seed, max_samples)
    elif max_samples == 100 and thumos==0:
        subset_indices = [2104, 1358, 4156, 2836, 4850, 3028, 4364, 1079, 2241, 3376, 4571, 1198, 4407, 4138, 418, 1548, 3448, 1096, 2094, 1608, 3771, 376, 4098, 3443, 3644, 2492, 2783, 2330, 1807, 3129, 4493, 4342, 
                          3219, 3591, 3400, 4810, 3542, 3811, 2218, 1833, 1487, 1027, 1088, 3765, 3154, 3677, 3915, 47, 773, 1632, 2356, 294, 4448, 2690, 649, 2428, 4213, 963, 4023, 1534, 775, 1335, 571, 1780, 2922, 
                          3197, 2541, 2261, 3931, 4323, 2425, 3078, 119, 968, 2845, 117, 1501, 4247, 4923, 4004, 1369, 2007, 3188, 4266, 831, 3402, 824, 1612, 296, 2942, 1889, 321, 1077, 695, 2245, 3711, 3859, 558, 1070, 221]
        dataset = CustomSubset(dataset, subset_indices)
    elif max_samples==100:
        # subset_indices = [178, 179, 322, 323, 324, 325, 326, 327, 328, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 616, 406, 407, 408, 409, 410, 411, 412, 413, 414, 
        #                   415, 416, 417, 418, 419, 420, 88, 89, 464, 465, 466, 467, 721, 722, 723, 354, 355, 682, 683, 403, 404, 405, 332, 620, 621, 622, 623, 624, 625, 437, 274, 275, 276, 277, 397, 398, 149, 150, 201, 202, 203, 
        #                   204, 205, 206, 61, 62, 63, 64, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 30, 122, 123, 
        #                   22, 705, 706, 707, 708, 65, 66, 238, 239, 240, 163, 164, 165, 166, 643, 644, 645, 646, 339, 340, 341, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 507, 508, 734, 735, 736, 737, 451, 
        #                   452, 453, 454, 455, 343, 344, 345, 346, 669, 670, 671, 672, 673, 674, 675, 421, 422, 423, 424, 425, 399, 400, 401, 402, 333, 334, 459, 627, 628, 629, 39, 40, 41, 278, 699, 700, 139, 140, 141, 
        #                   142, 143, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 658, 659, 730, 731, 732, 733, 263, 264, 329, 330, 331, 725, 684, 194, 195, 196, 265, 266, 267, 724, 433, 434, 435, 436, 579, 
        #                   580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 34, 35, 36, 255, 256, 257, 258, 86, 87, 439, 440, 441, 442, 81, 82, 77, 78, 79, 80, 236, 428, 429, 
        #                   430, 431, 432, 218, 738, 739, 136, 137, 138, 365, 167, 168, 169, 633, 634, 635, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
        #                   302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 676, 701, 702, 703, 231, 232, 392, 393, 394, 444, 445, 117, 118, 172, 740, 741, 180]
        random.seed(seed)
        class_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 25, 26, 27, 28, 29, 31, 32, 37, 38, 42, 43, 44, 45, 46, 47, 48, 74, 83, 84, 94, 95, 96, 97, 98, 124, 144, 145, 146]
        with open("video_to_indices.json", "r") as f:
            video_to_indices = json.load(f)
        video_names = list(video_to_indices.keys())
        random.shuffle(video_names)
        subset_indices = class_indices
        for video_name in video_names:
            if len(subset_indices) < max_samples:
                subset_indices += video_to_indices[video_name]
        print(f"subset_indices: {subset_indices}; length: {len(subset_indices)}")
        dataset = CustomSubset(dataset, subset_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    assert batch_size % world_size == 0, f"batch size {batch_size} should be divided by world size {world_size}"
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size // world_size,
        collate_fn=collate,
        pin_memory=True,
        sampler=sampler,
        **kwargs,
    )
    return dataloader


def collate(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    gpu_stack_keys = ["inputs", "masks"]

    collate_data = {}
    for key in batch[0]:
        if key in gpu_stack_keys:
            collate_data[key] = default_collate([sample[key] for sample in batch])
        else:
            collate_data[key] = [sample[key] for sample in batch]
    return collate_data


def get_class_index(gt_json_path, class_map_path):
    with open(gt_json_path, "r") as f:
        anno = json.load(f)

    anno = anno["database"]
    class_map = []
    for video_name in anno.keys():
        if "annotations" in anno[video_name]:
            for tmpp_data in anno[video_name]["annotations"]:
                if tmpp_data["label"] not in class_map:
                    class_map.append(tmpp_data["label"])

    class_map.sort()
    f2 = open(class_map_path, "w")
    for name in class_map:
        f2.write(name + "\n")
    f2.close()
    return class_map
