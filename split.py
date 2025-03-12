#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集标签拆分工具

将数据集标签文件拆分为训练集和验证集，支持自定义拆分比例和输入/输出文件路径。
默认情况下，将输入文件按指定比例拆分为训练集和验证集。
支持追加模式，可以将多个文件的处理结果追加到同一输出文件中。
"""

import os
import random
import argparse
from typing import List, Tuple, Dict, Optional
import logging


def setup_logger() -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger("split_dataset")
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    
    return logger


def read_file(file_path: str) -> List[str]:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件内容列表，每行作为一个元素
    """
    logger = logging.getLogger("split_dataset")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        logger.info(f"成功读取文件 {file_path}，共 {len(lines)} 行")
        return lines
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {str(e)}")
        raise


def write_file(file_path: str, lines: List[str], append: bool = False) -> None:
    """
    将内容写入文件
    
    Args:
        file_path: 输出文件路径
        lines: 要写入的内容列表
        append: 是否追加模式，默认为False
    """
    logger = logging.getLogger("split_dataset")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            f.write('\n'.join(lines))
            # 如果是追加模式且有内容，确保以换行符结尾
            if append and lines:
                f.write('\n')
        action = "追加到" if append else "写入"
        logger.info(f"成功{action}文件 {file_path}，共 {len(lines)} 行")
    except Exception as e:
        logger.error(f"写入文件 {file_path} 时出错: {str(e)}")
        raise


def split_data(data: List[str], train_ratio: float) -> Tuple[List[str], List[str]]:
    """
    将数据拆分为训练集和验证集
    
    Args:
        data: 要拆分的数据列表
        train_ratio: 训练集比例 (0.0-1.0)
        
    Returns:
        训练集和验证集的元组
    """
    # 复制数据并打乱顺序
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算训练集大小
    train_size = int(len(shuffled_data) * train_ratio)
    
    # 拆分数据
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]
    
    return train_data, val_data


def process_file(
    input_file: str, 
    train_output: str, 
    val_output: str, 
    train_ratio: float,
    seed: Optional[int] = None,
    append: bool = False
) -> Dict[str, int]:
    """
    处理单个文件的拆分
    
    Args:
        input_file: 输入文件路径
        train_output: 训练集输出文件路径
        val_output: 验证集输出文件路径
        train_ratio: 训练集比例 (0.0-1.0)
        seed: 随机种子
        append: 是否追加到输出文件，默认为False
        
    Returns:
        包含处理结果统计信息的字典
    """
    logger = logging.getLogger("split_dataset")
    
    if seed is not None:
        random.seed(seed)
    
    # 读取数据
    data = read_file(input_file)
    
    # 拆分数据
    train_data, val_data = split_data(data, train_ratio)
    
    # 写入文件
    write_file(train_output, train_data, append)
    write_file(val_output, val_data, append)
    
    # 返回统计信息
    stats = {
        "total": len(data),
        "train": len(train_data),
        "val": len(val_data),
        "train_ratio": train_ratio
    }
    
    logger.info(f"文件 {input_file} 处理完成:")
    logger.info(f"  总行数: {stats['total']}")
    logger.info(f"  训练集: {stats['train']} 行 ({stats['train_ratio']:.2f})")
    logger.info(f"  验证集: {stats['val']} 行 ({1-stats['train_ratio']:.2f})")
    
    return stats


def main():
    """主函数"""
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="数据集标签拆分工具")
    
    # 添加参数
    parser.add_argument("--label-file", type=str, required=True,
                        help="标签数据文件路径")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练集比例 (默认: 0.8)")
    parser.add_argument("--output-dir", type=str, default="output/split",
                        help="输出目录 (默认: output/split)")
    parser.add_argument("--train-output", type=str, default="train.txt",
                        help="训练集输出文件名 (默认: train.txt)")
    parser.add_argument("--val-output", type=str, default="val.txt",
                        help="验证集输出文件名 (默认: val.txt)")
    parser.add_argument("--append", action="store_true",
                        help="追加模式：将结果追加到现有输出文件")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细日志")
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志级别
    logger = setup_logger()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 设置输出文件路径
    train_output = os.path.join(args.output_dir, args.train_output)
    val_output = os.path.join(args.output_dir, args.val_output)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理文件
    logger.info("开始处理数据集拆分...")
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 处理文件
    if os.path.exists(args.label_file):
        stats = process_file(
            args.label_file, 
            train_output, 
            val_output, 
            args.train_ratio,
            args.seed,
            args.append
        )
    else:
        logger.error(f"文件 {args.label_file} 不存在，无法处理")
        return
    
    logger.info("数据集拆分处理完成")


if __name__ == "__main__":
    main() 