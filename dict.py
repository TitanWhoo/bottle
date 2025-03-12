#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字符提取工具

该脚本用于从文本文件中提取唯一字符并保存到字典文件中。
支持自定义输入和输出文件路径。
"""

import os
import sys
import argparse
import logging
from typing import Set, List


def setup_logging(verbose: bool = False) -> None:
    """
    设置日志配置
    
    Args:
        verbose: 是否启用详细日志
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def read_characters_from_file(input_path: str, delimiter: str = '\t', column_index: int = 1) -> Set[str]:
    """
    从文件中读取并提取所有唯一字符
    
    Args:
        input_path: 输入文件路径
        delimiter: 分隔符，默认为制表符
        column_index: 包含文本的列索引，默认为1（第二列）
    
    Returns:
        包含所有唯一字符的集合
    
    Raises:
        FileNotFoundError: 如果输入文件不存在
        IOError: 如果读取文件时出错
    """
    all_chars = set()
    line_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(delimiter)
                if len(parts) > column_index:
                    text = parts[column_index]
                    for char in text:
                        all_chars.add(char)
                    line_count += 1
                else:
                    logging.warning(f"第 {line_num} 行格式不正确，已跳过: {line}")
        
        logging.info(f"已处理 {line_count} 行文本")
        return all_chars
    
    except FileNotFoundError:
        logging.error(f"找不到输入文件: {input_path}")
        raise
    except IOError as e:
        logging.error(f"读取文件时出错: {e}")
        raise


def save_characters_to_file(characters: List[str], output_path: str) -> None:
    """
    将字符列表保存到文件
    
    Args:
        characters: 要保存的字符列表
        output_path: 输出文件路径
    
    Raises:
        IOError: 如果写入文件时出错
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"已创建输出目录: {output_dir}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for char in characters:
                f.write(char + '\n')
        
        logging.info(f"已将 {len(characters)} 个唯一字符保存到: {output_path}")
    
    except IOError as e:
        logging.error(f"写入文件时出错: {e}")
        raise


def extract_characters_to_dict(input_path: str, output_path: str, delimiter: str = '\t', column_index: int = 1) -> None:
    """
    从输入文件中提取唯一字符并保存到输出文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        delimiter: 分隔符，默认为制表符
        column_index: 包含文本的列索引，默认为1（第二列）
    """
    logging.info(f"开始从 {input_path} 提取字符")
    
    # 读取并提取字符
    all_chars = read_characters_from_file(input_path, delimiter, column_index)
    
    # 将字符集合转换为排序列表
    char_list = sorted(list(all_chars))
    logging.info(f"共提取了 {len(char_list)} 个唯一字符")
    
    # 保存到输出文件
    save_characters_to_file(char_list, output_path)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='从文本文件中提取唯一字符并保存到字典文件')
    
    parser.add_argument('-i', '--input', 
                        default='./datasets/rec/all.txt',
                        help='输入文件路径，默认为 ./datasets/rec/all.txt')
    
    parser.add_argument('-o', '--output', 
                        default='./dict.txt',
                        help='输出文件路径，默认为 ./dict.txt')
    
    parser.add_argument('-d', '--delimiter', 
                        default='\t',
                        help='输入文件的分隔符，默认为制表符')
    
    parser.add_argument('-c', '--column', 
                        type=int,
                        default=1,
                        help='包含文本的列索引（从0开始），默认为1（第二列）')
    
    parser.add_argument('-v', '--verbose', 
                        action='store_true',
                        help='启用详细日志输出')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        extract_characters_to_dict(
            input_path=args.input,
            output_path=args.output,
            delimiter=args.delimiter,
            column_index=args.column
        )
        logging.info("处理完成")
        return 0
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())