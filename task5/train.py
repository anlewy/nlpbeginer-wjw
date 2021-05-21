from tqdm import tqdm
import torch
import torch.nn as nn
import math

LEARNING_RATE = 0.1
DECAY_RATE = 0.1


def evaluate(model, loss_func, data_iter, is_dev=False, epoch=None):
    model.eval()
    total_words = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            text, lens = batch.text

            inputs = text[:, :-1]
            targets = text[:, 1:]
            init_hidden = model.lstm.init_hidden(inputs.size(0))
            logits, _ = model(text, lens - 1, init_hidden)
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            total_loss += loss.item()
            total_words += lens.sum().item()
    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s perplexity %.3f" % (
                epoch + 1, "Dev" if is_dev else "Test", math.exp(total_loss / total_words)
            )
        )
    else:
        tqdm.write(
            "%s perplexity %.3f" % ("Dev" if is_dev else "Test", math.exp(total_loss / total_words))
        )


def train(model, loss_func, optimizer, train_iter, valid_iter, epochs=10, clip=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_words = 0
        for i, batch in enumerate(tqdm(train_iter)):
            text, lens = batch.text
            inputs = text[:, :-1]
            targets = text[:, 1:]
            init_hidden = model.lstm.init_hidden(inputs.size(0))
            logits, _ = model(inputs, lens - 1, init_hidden)
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss
            total_words += lens.sum().item()
        tqdm.write("Epoch %d, Train perplexity: %d" % (epoch + 1, math.exp(total_loss / total_words)))
        evaluate(model, loss_func, data_iter=valid_iter, is_dev=True, epoch=epoch)

        # 学习率衰减
        lr = LEARNING_RATE / (1 + (epoch + 1) * DECAY_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def generate(model, TEXT, eos_idx, word, temperature=0.8, device='cpu'):
    model.eval()
    with torch.no_grad():
        if word in TEXT.vocab.stoi:
            idx = TEXT.vocab.stoi[word]
            inputs = torch.tensor([idx])
        else:
            print("%s is not in vocabulary, choose by random." % word)
            prob = torch.ones(len(TEXT.vocab))
            inputs = torch.multinomial(prob, 1)
            idx = inputs[0].item()

        inputs = inputs.unsqueeze(1).to(device)
        lens = torch.tensor([1]).to(device)
        hidden = tuple([h.to(device) for h in model.lstm.init_hidden(1)])
        poetry = [TEXT.vocab.itos[idx]]

        while idx != eos_idx:
            logits, hidden = model(inputs, lens, hidden)
            word_weights = logits.squeeze().div(temperature).exp().cpu()
            idx = torch.multinomial(word_weights, 1)[0].item()
            inputs.fill_(idx)
            poetry.append(TEXT.vocab.itos[idx])
        print(''.join(poetry[:-1]))
