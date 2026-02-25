#!/usr/bin/env python3
"""
CHAIR 评估脚本
使用 DeCo 的 chair.py 实现 CHAIR 指标（CHAIRs、CHAIRi），同时输出 Recall 与 Len。

依赖：
- /data/home/scyb531/DeCo/chair.py（本地拷贝的 CHAIR 评测实现）
- NLTK（分词、词性标注、WordNet 词形还原）；请设置 NLTK_DATA 环境变量
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

# 添加 DeCo 路径并导入 CHAIR 模块
DECO_ROOT = "/data/home/scyb531/DeCo"
if DECO_ROOT not in sys.path:
    sys.path.insert(0, DECO_ROOT)

try:
    # DeCo/chair.py 提供 CHAIR 类与 CLI；此处直接复用 CHAIR 类
    from chair import CHAIR  # type: ignore
except Exception as e:
    print(f"[ERROR] 无法导入 DeCo 的 chair.py：{e}")
    sys.exit(1)


def read_jsonl(file_path: str):
    """读取 JSONL 文件"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def parse_coco_int_id(image_id) -> int:
    """
    从字符串中提取 COCO image_id（整数）
    
    兼容格式：
    - "COCO_val2014_000000395849"
    - "000000395849"
    - "395849"
    - 395849（int）
    """
    if isinstance(image_id, int):
        return image_id
    
    s = str(image_id)
    # 优先匹配末尾 6-12 位数字
    m = re.search(r"(\d{6,12})$", s)
    if m:
        return int(m.group(1))
    
    # 回退到任意数字
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    
    raise ValueError(f"无法解析 COCO image_id: {image_id}")


def compute_bleu_metrics(jsonl_file: str, coco_annotations_dir: str) -> Dict[str, float]:
    """
    计算 BLEU 分数（使用 NLTK 或 pycocoevalcap）
    
    Args:
        jsonl_file: 预测 JSONL 文件
        coco_annotations_dir: COCO annotations 目录
        
    Returns:
        Dict[str, float]: 包含 Bleu_1, Bleu_2, Bleu_3, Bleu_4 和平均 BLEU 的字典
    """
    # 首先尝试使用 pycocoevalcap
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
        
        # 读取预测结果
        pred_data = read_jsonl(jsonl_file)
        
        # 构建预测结果列表（COCO格式）
        pred_list = []
        for item in pred_data:
            img_id = parse_coco_int_id(item["image_id"])
            caption = item.get("caption", "").strip()
            if caption:
                pred_list.append({
                    "image_id": img_id,
                    "caption": caption
                })
        
        # 查找 COCO annotations 文件
        import glob
        ann_files = glob.glob(os.path.join(coco_annotations_dir, "captions_*2014.json"))
        if not ann_files:
            print(f"[WARNING] 未找到 COCO captions 文件，尝试使用 NLTK 计算 BLEU")
            raise FileNotFoundError("COCO annotations not found")
        
        # 使用第一个找到的 annotations 文件
        ann_file = ann_files[0]
        print(f"[INFO] 使用 COCO annotations: {ann_file}")
        coco = COCO(ann_file)
        cocoRes = coco.loadRes(pred_list)
        
        # 计算 BLEU（类似 sjc/Nullu 的实现）
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params["image_id"] = cocoRes.getImgIds()
        cocoEval.evaluate()
        
        # 提取 BLEU_1 到 BLEU_4 分数
        bleu_result = {
            "Bleu_1": cocoEval.eval.get("Bleu_1", 0.0),
            "Bleu_2": cocoEval.eval.get("Bleu_2", 0.0),
            "Bleu_3": cocoEval.eval.get("Bleu_3", 0.0),
            "Bleu_4": cocoEval.eval.get("Bleu_4", 0.0),
        }
        
        # 计算平均 BLEU（与 sjc/Nullu 一致）
        bleu_result["BLEU_avg"] = (
            bleu_result["Bleu_1"] + 
            bleu_result["Bleu_2"] + 
            bleu_result["Bleu_3"] + 
            bleu_result["Bleu_4"]
        ) / 4.0
        
        print(f"[INFO] BLEU 计算完成: Bleu_1={bleu_result['Bleu_1']:.4f}, "
              f"Bleu_2={bleu_result['Bleu_2']:.4f}, "
              f"Bleu_3={bleu_result['Bleu_3']:.4f}, "
              f"Bleu_4={bleu_result['Bleu_4']:.4f}, "
              f"平均={bleu_result['BLEU_avg']:.4f}")
        
        return bleu_result
            
    except (ImportError, FileNotFoundError, ValueError) as e:
        # 回退到使用 NLTK 计算 BLEU
        print(f"[INFO] pycocoevalcap 不可用 ({e})，使用 NLTK 计算 BLEU")
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            import nltk
            
            # 确保 NLTK 数据已下载
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("[INFO] 下载 NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            # 读取预测结果和参考 captions
            pred_data = read_jsonl(jsonl_file)
            
            # 读取 COCO annotations
            import glob
            import json
            ann_files = glob.glob(os.path.join(coco_annotations_dir, "captions_*2014.json"))
            if not ann_files:
                print(f"[WARNING] 未找到 COCO captions 文件，跳过 BLEU 计算")
                return {
                    "Bleu_1": 0.0, "Bleu_2": 0.0, 
                    "Bleu_3": 0.0, "Bleu_4": 0.0, 
                    "BLEU_avg": 0.0
                }
            
            with open(ann_files[0], 'r') as f:
                coco_ann = json.load(f)
            
            # 构建 image_id -> captions 映射
            img_to_refs = {}
            for ann in coco_ann.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in img_to_refs:
                    img_to_refs[img_id] = []
                img_to_refs[img_id].append(word_tokenize(ann['caption'].lower()))
            
            # 计算每个预测的 BLEU-1, 2, 3, 4 并取平均
            smoothing = SmoothingFunction().method1
            bleu_1_scores, bleu_2_scores = [], []
            bleu_3_scores, bleu_4_scores = [], []
            
            for item in pred_data:
                img_id = parse_coco_int_id(item["image_id"])
                pred_caption = item.get("caption", "").strip()
                if not pred_caption or img_id not in img_to_refs:
                    continue
                
                pred_tokens = word_tokenize(pred_caption.lower())
                refs = img_to_refs[img_id]
                
                # 计算不同 n-gram 的 BLEU 分数
                for ref in refs:
                    bleu_1_scores.append(sentence_bleu([ref], pred_tokens, smoothing_function=smoothing, weights=(1.0, 0, 0, 0)))
                    bleu_2_scores.append(sentence_bleu([ref], pred_tokens, smoothing_function=smoothing, weights=(0.5, 0.5, 0, 0)))
                    bleu_3_scores.append(sentence_bleu([ref], pred_tokens, smoothing_function=smoothing, weights=(0.33, 0.33, 0.33, 0)))
                    bleu_4_scores.append(sentence_bleu([ref], pred_tokens, smoothing_function=smoothing, weights=(0.25, 0.25, 0.25, 0.25)))
            
            if bleu_1_scores:
                bleu_result = {
                    "Bleu_1": sum(bleu_1_scores) / len(bleu_1_scores),
                    "Bleu_2": sum(bleu_2_scores) / len(bleu_2_scores),
                    "Bleu_3": sum(bleu_3_scores) / len(bleu_3_scores),
                    "Bleu_4": sum(bleu_4_scores) / len(bleu_4_scores),
                }
                bleu_result["BLEU_avg"] = (
                    bleu_result["Bleu_1"] + bleu_result["Bleu_2"] + 
                    bleu_result["Bleu_3"] + bleu_result["Bleu_4"]
                ) / 4.0
                return bleu_result
            else:
                return {
                    "Bleu_1": 0.0, "Bleu_2": 0.0, 
                    "Bleu_3": 0.0, "Bleu_4": 0.0, 
                    "BLEU_avg": 0.0
                }
                
        except Exception as e2:
            print(f"[WARNING] NLTK BLEU 计算也失败: {e2}，跳过 BLEU 计算")
            return {
                "Bleu_1": 0.0, "Bleu_2": 0.0, 
                "Bleu_3": 0.0, "Bleu_4": 0.0, 
                "BLEU_avg": 0.0
            }


def compute_chair_metrics(jsonl_file: str, coco_annotations_dir: str, cache_file: str = "") -> Dict[str, Any]:
    """
    使用 DeCo 的 CHAIR 实现计算指标。

    Args:
        jsonl_file: 预测 JSONL 文件（每行包含 image_id, caption）
        coco_annotations_dir: COCO annotations 目录（需包含 captions_*2014.json 与 instances_*2014.json）
        cache_file: 可选，CHAIR 评估器缓存（pickle）路径

    Returns:
        dict: 与 DeCo/chair.py 一致的输出（sentences 与 overall_metrics）
    """
    import pickle

    if cache_file and os.path.exists(cache_file):
        try:
            evaluator = pickle.load(open(cache_file, 'rb'))
            print(f"loaded evaluator from cache: {cache_file}")
        except Exception as e:
            print(f"缓存加载失败（将重建）：{e}")
            evaluator = CHAIR(coco_annotations_dir)
            if cache_file:
                pickle.dump(evaluator, open(cache_file, 'wb'))
                print(f"cached evaluator to: {cache_file}")
    else:
        print("cache not set or not exist yet, building from scratch...")
        evaluator = CHAIR(coco_annotations_dir)
        if cache_file:
            pickle.dump(evaluator, open(cache_file, 'wb'))
            print(f"cached evaluator to: {cache_file}")

    cap_dict = evaluator.compute_chair(jsonl_file, "image_id", "caption")
    return cap_dict


# 旧版 compute_chair_metrics（依赖 COCOEvalCap）已移除；转而直接调用 DeCo 的 CHAIR。


def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR 评估")
    parser.add_argument("--input", type=str, required=True, 
                       help="生成的 caption JSONL 文件")
    parser.add_argument("--coco-annotations", type=str, 
                       default="/data/home/scyb531/DATA/annotations",
                       help="COCO annotations 目录（包含 captions/instances *2014.json）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出 JSON 文件（默认自动生成）")
    parser.add_argument("--cache", type=str, default="/data/home/scyb531/DeCo/eval_Nullu/CHAIR/chair.pkl",
                       help="CHAIR 评估器缓存（pickle）路径；留空则不使用缓存")
    parser.add_argument("--verbose", action="store_true",
                       help="打印详细结果")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}")
        sys.exit(1)
    
    print("=" * 80)
    print("CHAIR 评估")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"COCO Annotations: {args.coco_annotations}")
    print("=" * 80)
    
    # 计算 CHAIR 指标（DeCo 实现）
    chair_result = compute_chair_metrics(
        jsonl_file=args.input,
        coco_annotations_dir=args.coco_annotations,
        cache_file=args.cache if args.cache else "",
    )
    
    # 汇总 per-image 结果
    halc_caption_result = chair_result.get("sentences", [])
    halc_result: Dict[int, Dict[str, Any]] = {}
    for item in halc_caption_result:
        img_id = item["image_id"]
        metrics = item.get("metrics", {})
        halc_result[img_id] = {
            "caption": item.get("caption", ""),
            "chairs": metrics.get("CHAIRs", 0),
            "chairi": metrics.get("CHAIRi", 0.0),
            "recall": metrics.get("Recall", 0.0),
            "objects_num": len(item.get("mscoco_generated_words", [])),
            "words_num": len(item.get("words", [])),
            "hallucinate_num": len(item.get("hallucination_idxs", [])),
        }
    
    overall = chair_result.get("overall_metrics", {})
    chairs_avg = float(overall.get("CHAIRs", 0.0))
    chairi_avg = float(overall.get("CHAIRi", 0.0))
    recall_avg = float(overall.get("Recall", 0.0))
    len_avg = float(overall.get("Len", 0.0))
    n_samples = len(halc_result)
    
    # 计算 BLEU（与 sjc/Nullu 一致的方式）
    print("\n计算 BLEU 分数...")
    bleu_metrics = compute_bleu_metrics(args.input, args.coco_annotations)
    
    # 将 BLEU 分数添加到 overall_metrics
    overall.update(bleu_metrics)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("CHAIR 评估结果")
    print("=" * 80)
    print(f"样本数:          {n_samples}")
    print(f"CHAIRs (平均):   {chairs_avg:.4f}")
    print(f"CHAIRi (平均):   {chairi_avg:.4f}")
    print(f"Recall (平均):    {recall_avg:.4f}")
    print(f"Len (平均):       {len_avg:.4f}")
    print(f"BLEU_1:           {bleu_metrics['Bleu_1']:.4f}")
    print(f"BLEU_2:           {bleu_metrics['Bleu_2']:.4f}")
    print(f"BLEU_3:           {bleu_metrics['Bleu_3']:.4f}")
    print(f"BLEU_4:           {bleu_metrics['Bleu_4']:.4f}")
    print(f"BLEU_avg:         {bleu_metrics['BLEU_avg']:.4f}")
    print("=" * 80)
    
    # 详细结果（可选）
    if args.verbose:
        print("\n每张图像的详细结果:")
        for img_id, res in sorted(halc_result.items()):
            print(f"\nImage {img_id}:")
            print(f"  Caption: {res['caption']}")
            print(f"  CHAIRs: {res['chairs']:.4f}, CHAIRi: {res['chairi']:.4f}, Recall: {res['recall']:.4f}")
            print(f"  Objects: {res['objects_num']}, Hallucinations: {res['hallucinate_num']}")
    
    # 保存结果
    if args.output:
        output_file = args.output
    else:
        output_file = args.input.replace(".jsonl", "_chair_result.json")
    
    result = {
        "input_file": args.input,
        "num_samples": n_samples,
        "overall_metrics": overall,  # 已包含 BLEU
        "per_image_results": halc_result,
        "full_chair_output": chair_result
    }
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

