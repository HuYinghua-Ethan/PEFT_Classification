# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model",
    "train_data_path": "data/train_tag_news.json",
    "valid_data_path": "data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epochs": 10,
    "batch_size": 64,
    "tuning_tactics":"lora_tuning",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\2024 AILearning\bert-base-chinese",
    "seed": 987
}