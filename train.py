from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from common import params
from torch import nn, optim
from transformer import (
    build_model,
    device,
    infinite_iter,
    save_model,
    schedule_sampling,
)
import numpy as np
from read_file import PCRJJCDataset


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_iter,
    loss_function,
    total_steps,
    summary_steps,
    train_dataset,
):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling(step, summary_steps))
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print(
                "\r",
                "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(
                    total_steps + step + 1, loss_sum, np.exp(loss_sum)
                ),
                end=" ",
            )
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses


def test(model: nn.Module, dataloader: PCRJJCDataset, loss_function):
    model.eval()
    loss_sum = 0.0
    for sources, targets in dataloader:
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model.inference(sources, targets)
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)

        loss = loss_function(outputs.to(device), targets)
        loss_sum += loss.item()

    return loss_sum / len(dataloader)


def train_process(config):
    # 準備訓練資料

    train_dataset = PCRJJCDataset(
        Path(__file__).parent / "dataset" / "orginal_train.json"
    )
    dict_len = train_dataset.dict_len
    train_size = int(params.train_size_percentage * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 準備檢驗資料
    val_loader = DataLoader(val_dataset, batch_size=120)
    # 建構模型
    model, optimizer = build_model(config, dict_len)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses = [], []
    total_steps = 0
    while total_steps < config.num_steps:
        # 訓練模型
        model, optimizer, loss = train(
            model,
            optimizer,
            train_iter,
            loss_function,
            total_steps,
            config.summary_steps,
            train_dataset,
        )
        train_losses += loss
        # 檢驗模型
        val_loss = test(model, val_loader, loss_function)
        val_losses.append(val_loss)

        total_steps += config.summary_steps
        print(
            "\r",
            "val [{}] loss: {:.3f}, Perplexity: {:.3f}".format(
                total_steps, val_loss, np.exp(val_loss)
            ),
        )

        # 儲存模型和結果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, optimizer, config.store_model_path, total_steps)

    return train_losses, val_losses


def test_process(config):
    # 準備測試資料
    test_dataset = PCRJJCDataset(
        Path(__file__).parent / "dataset" / "orginal_test.json"
    )
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 建構模型
    model, optimizer = build_model(config, test_dataset.dict_len)
    print("Finish build model")
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    # 測試模型
    test_loss = test(model, test_loader, loss_function)

    return test_loss


def use_process2(config):
    test_dataset = PCRJJCDataset(Path(__file__).parent / "dataset" / "use.json")
    test_loader = DataLoader(test_dataset, batch_size=1)
    model, optimizer = build_model(config, test_dataset.dict_len)
    model.eval()
    for sources, targets in test_loader:
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model.inference(sources, targets)
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)

    return outputs, preds


def use_process(config):
    test_dataset = PCRJJCDataset(Path(__file__).parent / "dataset" / "use.json")
    test_loader = DataLoader(test_dataset, batch_size=1)
    model, optimizer = build_model(config, test_dataset.dict_len)
    model.eval()
    with torch.no_grad():
        for sources, targets in test_loader:
            sources, targets = sources.to(device), targets.to(device)
            outputs, preds = model(sources, targets, 0)

    return preds
