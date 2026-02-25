"""
CHAIR 数据集构建器
参考 Nullu 的采样方式，确保与其他方法的评估一致性
"""
import os
import random
from typing import List, Dict, Any
from pycocotools.coco import COCO


class CHAIRDataset:
    """CHAIR 数据集构建器（参考 Nullu/dataset/CHAIR.py）"""
    
    def __init__(
        self, 
        split: str = "val", 
        data_root: str = "/data/home/scyb531/DATA/", 
        sampling: str = "random", 
        num_samples: int = 500,
        seed: int = 0
    ):
        """
        Args:
            split: "val" 或 "train"
            data_root: COCO 数据根目录
            sampling: "first" 或 "random"
            num_samples: 采样数量
            seed: 随机种子
        """
        self.split = split
        self.ann_path = os.path.join(data_root, f"annotations/instances_{split}2014.json")
        self.caption_path = os.path.join(data_root, f"annotations/captions_{split}2014.json")
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.sampling = sampling
        self.num_samples = num_samples
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
    
    def get_data(self) -> List[Dict[str, Any]]:
        """
        获取数据列表
        
        Returns:
            List[Dict]: 每个元素包含 image_id, image_path, question
        """
        coco = COCO(self.caption_path)
        
        img_ids = coco.getImgIds()
        
        # 采样策略
        if self.num_samples:
            if self.num_samples > len(img_ids):
                print(f"[WARN] num_samples {self.num_samples} 超过图像总数 ({len(img_ids)})")
                sampled_img_ids = img_ids
            elif self.sampling == "first":
                sampled_img_ids = img_ids[:self.num_samples]
            elif self.sampling == "random":
                sampled_img_ids = random.sample(img_ids, self.num_samples)
            else:
                raise ValueError(f"不支持的采样策略: {self.sampling}")
        else:
            sampled_img_ids = img_ids
        
        # 构建数据列表
        val_data = []
        for cur_img_id in sampled_img_ids:
            cur_img = coco.loadImgs(cur_img_id)[0]
            cur_img_path = cur_img["file_name"]
            val_data.append({
                "image_id": cur_img_id,
                "image_path": os.path.join(self.img_root, cur_img_path),
                "question": "Please describe this image in detail."
            })
        
        return val_data


def build_chair_dataset(
    split: str = "val",
    data_root: str = "/data/home/scyb531/DATA/",
    sampling: str = "random",
    num_samples: int = 500,
    seed: int = 0
) -> List[Dict[str, Any]]:
    """
    构建 CHAIR 数据集（便捷函数）
    
    Returns:
        List[Dict]: 数据列表
    """
    dataset = CHAIRDataset(split, data_root, sampling, num_samples, seed)
    return dataset.get_data()

