#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def read_file(file_path):
    """
    读取文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        sys.exit(1)


def preprocess_text(text):
    """
    文本预处理：去除标点符号、分词
    """
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 使用jieba进行分词
    words = jieba.cut(text)
    return ' '.join(words)


def calculate_similarity(orig_text, copy_text):
    """
    计算文本相似度
    """
    # 预处理文本
    orig_processed = preprocess_text(orig_text)
    copy_processed = preprocess_text(copy_text)

    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer()

    # 将文本转换为TF-IDF向量
    tfidf_matrix = vectorizer.fit_transform([orig_processed, copy_processed])

    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return round(similarity, 2)


def write_result(result_file, similarity):
    """
    将结果写入文件
    """
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"文本相似度: {similarity}")
        print(f"结果已写入文件: {result_file}")
    except Exception as e:
        print(f"写入结果文件时出错: {e}")
        sys.exit(1)


def main():
    """
    主函数
    """
    # 检查参数数量
    if len(sys.argv) != 4:
        print("用法: python 论文查重.py [原文文件] [抄袭版论文的文件] [答案文件]")
        sys.exit(1)

    # 获取文件路径
    orig_file = sys.argv[1]
    copy_file = sys.argv[2]
    result_file = sys.argv[3]

    # 检查文件是否存在
    if not os.path.exists(orig_file):
        print(f"错误: 原文文件 {orig_file} 不存在")
        sys.exit(1)

    if not os.path.exists(copy_file):
        print(f"错误: 抄袭版论文文件 {copy_file} 不存在")
        sys.exit(1)

    # 读取文件内容
    orig_text = read_file(orig_file)
    copy_text = read_file(copy_file)

    # 计算相似度
    similarity = calculate_similarity(orig_text, copy_text)

    # 输出结果
    print(f"文本相似度: {similarity}")

    # 将结果写入文件
    write_result(result_file, similarity)


if __name__ == "__main__":
    main()