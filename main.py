# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import torch.nn as nn
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig 


#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

# 保存微调参数
def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel
    # 大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    
    model = get_peft_model(model, peft_config)
    # print(model.state_dict().keys())
    if tuning_tactics == "lora_tuning":
        """
        lora配置会冻结原始模型中的所有层，不允许其反传梯度
        但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        如果训练大模型，这段代码可以不用写，因为大部分的大模型都是生成式模型
        """
        for param in model.get_submodule("model").get_submodule("classifier").parameters():
            param.requires_grad = True

    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("CUDA is available")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载测试效果类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epochs"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data
            output = model(input_ids)[0]
            loss = nn.CrossEntropyLoss()(output, labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)  #保存模型权重
    return acc



if __name__ == "__main__":
    main(Config)

