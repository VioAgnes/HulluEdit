#!/usr/bin/env python3
"""
快速运行 CHAIR 评估的脚本
使用方法: python run_chair_eval.py [--input INPUT_JSONL] [其他参数]
"""
import argparse
import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入 eval_chair 模块
from eval_chair import compute_chair_metrics


def main():
    parser = argparse.ArgumentParser(description="运行 CHAIR 评估")
    parser.add_argument("--input", type=str, 
                       default="/data/home/scyb531/lyg/HulluEdit/outputs/chair/chair_captions_ecse_val_20251107_150256.jsonl",
                       help="输入的 caption JSONL 文件")
    parser.add_argument("--coco-annotations", type=str, 
                       default="/data/home/scyb531/DATA/annotations",
                       help="COCO annotations 目录")
    parser.add_argument("--output", type=str, default=None,
                       help="输出 JSON 文件（默认自动生成）")
    parser.add_argument("--cache", type=str, 
                       default="/data/home/scyb531/DeCo/eval_Nullu/CHAIR/chair.pkl",
                       help="CHAIR 评估器缓存路径")
    parser.add_argument("--verbose", action="store_true",
                       help="打印详细结果")
    
    args = parser.parse_args()
    
    # 设置 NLTK 数据路径
    os.environ["NLTK_DATA"] = "/data/home/scyb531/nltk_data"
    
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
    
    # 计算 CHAIR 指标
    chair_result = compute_chair_metrics(
        jsonl_file=args.input,
        coco_annotations_dir=args.coco_annotations,
        cache_file=args.cache if args.cache else "",
    )
    
    # 汇总结果
    halc_caption_result = chair_result.get("sentences", [])
    halc_result = {}
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
    
    # 打印结果
    print("\n" + "=" * 80)
    print("CHAIR 评估结果")
    print("=" * 80)
    print(f"样本数:          {n_samples}")
    print(f"CHAIRs (平均):   {chairs_avg:.4f}")
    print(f"CHAIRi (平均):   {chairi_avg:.4f}")
    print(f"Recall (平均):    {recall_avg:.4f}")
    print(f"Len (平均):       {len_avg:.4f}")
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
    
    import json
    result = {
        "input_file": args.input,
        "num_samples": n_samples,
        "overall_metrics": overall,
        "per_image_results": halc_result,
        "full_chair_output": chair_result
    }
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

