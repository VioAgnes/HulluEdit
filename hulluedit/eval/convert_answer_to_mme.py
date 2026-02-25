"""
将 MME 答案转换为 MME 官方评测格式
参考 DeCo/eval_tool/convert_answer_to_mme.py 的实现
"""
import os
import json
import argparse
from collections import defaultdict
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="将 MME 答案转换为官方评测格式")
    parser.add_argument('--output_path', type=str, required=True,
                        help="模型输出的答案文件路径（JSONL 格式）")
    parser.add_argument('--seed', default=0, type=int,
                        help="随机种子")
    parser.add_argument('--log_path', type=str, required=True,
                        help="转换后的结果保存目录")
    parser.add_argument('--data_path', type=str, 
                        default="/data/home/scyb531/DATA/MME_Benchmark_release_version",
                        help="MME 数据集根目录（用于获取 ground truth）")
    args = parser.parse_args()
    return args


def get_gt(data_path):
    """
    从 MME 数据集中读取 ground truth
    参考 DeCo/eval_tool/convert_answer_to_mme.py
    """
    GT = {}
    data_path = Path(data_path).expanduser()
    
    if not data_path.exists():
        print(f"[WARNING] MME 数据集路径不存在: {data_path}")
        return GT
    
    for category in os.listdir(data_path):
        category_dir = data_path / category
        if not category_dir.is_dir():
            continue
        
        # 检查目录结构
        if (category_dir / 'images').exists():
            image_path = category_dir / 'images'
            qa_path = category_dir / 'questions_answers_YN'
        else:
            image_path = qa_path = category_dir
        
        if not image_path.is_dir():
            continue
        if not qa_path.is_dir():
            continue
        
        # 读取问题和答案
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            
            qa_file = qa_path / file
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                question = parts[0]
                                answer = parts[1]
                                GT[(category, file, question)] = answer
                        except Exception as e:
                            print(f"[WARNING] 解析行失败: {line}, 错误: {e}")
            except Exception as e:
                print(f"[WARNING] 读取文件失败: {qa_file}, 错误: {e}")
    
    return GT


def main():
    args = get_args()
    
    # 读取 ground truth
    print(f"[INFO] 读取 ground truth: {args.data_path}")
    GT = get_gt(args.data_path)
    print(f"[INFO] 共加载 {len(GT)} 个 ground truth 条目")
    
    # 创建输出目录
    result_dir = Path(args.log_path).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 结果将保存到: {result_dir}")
    
    # 读取模型答案
    print(f"[INFO] 读取模型答案: {args.output_path}")
    answers = []
    with open(args.output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                answers.append(json.loads(line))
            except Exception as e:
                print(f"[WARNING] 解析答案失败: {line}, 错误: {e}")
    
    print(f"[INFO] 共加载 {len(answers)} 个模型答案")
    
    # 按类别分组
    results = defaultdict(list)
    for answer in answers:
        question_id = answer.get('question_id', '')
        if not question_id:
            continue
        
        # 解析 question_id: "category/image_name.png"
        parts = question_id.split('/')
        if len(parts) < 2:
            print(f"[WARNING] 无效的 question_id: {question_id}")
            continue
        
        category = parts[0]
        image_name = parts[-1]
        file = image_name.rsplit('.', 1)[0] + '.txt'
        
        prompt = answer.get('prompt', '')
        text = answer.get('text', '')
        
        results[category].append((file, prompt, text))
    
    print(f"[INFO] 共 {len(results)} 个类别")
    
    # 写入转换后的结果
    total_matched = 0
    total_unmatched = 0
    
    for category, cate_tups in results.items():
        output_file = result_dir / f'{category}.txt'
        matched = 0
        unmatched = 0
        
        with open(output_file, 'w', encoding='utf-8') as fp:
            for file, prompt, answer in cate_tups:
                # 清理 prompt
                if 'Answer the question using a single word or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                
                # 尝试匹配 ground truth
                gt_ans = None
                
                # 尝试不同的 prompt 格式
                prompt_variants = [
                    prompt,
                    prompt + ' Please answer yes or no.',
                    prompt + '  Please answer yes or no.',  # 两个空格
                ]
                
                for p in prompt_variants:
                    if (category, file, p) in GT:
                        gt_ans = GT[(category, file, p)]
                        matched += 1
                        break
                
                if gt_ans is None:
                    # 未找到 ground truth
                    gt_ans = "UNKNOWN"
                    unmatched += 1
                
                # 写入结果：file \t prompt \t gt_ans \t answer
                tup = (file, prompt, gt_ans, answer)
                fp.write('\t'.join(tup) + '\n')
        
        total_matched += matched
        total_unmatched += unmatched
        print(f"[INFO] {category}: {matched} 匹配, {unmatched} 未匹配")
    
    print(f"\n[INFO] 转换完成！")
    print(f"[INFO] 总计: {total_matched} 匹配, {total_unmatched} 未匹配")
    print(f"[INFO] 结果保存在: {result_dir}")
    
    # 统计信息
    print(f"\n[INFO] 各类别文件:")
    for category_file in sorted(result_dir.glob('*.txt')):
        print(f"  - {category_file.name}")


if __name__ == "__main__":
    main()

