from tqdm import tqdm
import torch


def evaluate(model, loss_func, data_iter):
    model.eval()
    correct_num = 0
    error_num = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            premises, premise_lens = batch.premise
            hypotheses, hypothesis_lens = batch.hypothesis
            labels = batch.label

            output = model(premises, premise_lens, hypotheses, hypothesis_lens)
            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            error_num += (predicts != labels).sum().item()

        acc = correct_num / (correct_num + error_num)
        return acc


def train(model, loss_func, optimizer, train_iter, valid_iter, epochs, patience=5, clip=5):
    print("training...")
    return
    best_acc = -1
    patience_cnt = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_iter):
            premises, premise_lens = batch.premise
            hypotheses, hypothesis_lens = batch.hypothesis
            labels = batch.label

            model.zero_grad()
            output = model(premises, premise_lens, hypotheses, hypothesis_lens)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch + 1, total_loss))

        acc = evaluate(model, loss_func, valid_iter)
        if acc <= best_acc:
            patience_cnt += 1
        else:
            best_acc = acc
            patience_cnt = 0
            torch.save(model.state_dict(), "best_model.ckpt")
        if patience_cnt >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break
