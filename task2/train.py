import torch.nn as nn
from tqdm import tqdm
import torch


def evaluate(model, loss_func, data_iter):
    model.eval()
    correct_num = 0
    wrong_num = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            phrases, lens = batch.phrase
            labels = batch.sentiment

            output = model(phrases, lens)
            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()

            correct_num += (predicts == labels).sum().item()
            wrong_num += (predicts != labels).sum().item()

        acc = correct_num / (correct_num + wrong_num)
        return acc


def train(model, loss_func, optimizer, train_iter, valid_iter, epochs=16, patience=3, clip=5):
    best_acc = -1
    patience_cnt = 0

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        model.eval()
        for i, batch in enumerate(tqdm(train_iter)):
            phrases, lens = batch.phrase
            labels = batch.sentiment

            output = model(phrases, lens)
            loss = loss_func(output, labels)
            total_loss += loss.item()

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch+1, total_loss))

        acc = evaluate(model, loss_func, valid_iter)
        if acc > best_acc:
            best_acc = acc
            patience_cnt = 0
            torch.save(model.state_dict(), "model/best_model.ckpt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                tqdm.write("Early stopping: patience limit reached, stopping...")
                break
