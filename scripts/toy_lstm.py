from scripts.pums_downloader import datasets
from scripts.pums_sgd import main, ModelCoordinator, printf
from opendp.smartnoise.network.optimizer import PrivacyAccountant

import torch.nn as nn
import torch

import numpy as np


class LstmModule(nn.Module):

    def __init__(self, embedding_size, internal_size, vocab_size, tagset_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = internal_size
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, internal_size)
        self.hidden2tag = nn.Linear(internal_size, tagset_size)
        self.lstm_out = []
        self.lstm_hidden = []

    def forward(self, x):
        embed = self.embedding(x)
        hidden = self._init_hidden()

        # the second dimension refers to the batch size, which we've hard-coded
        # it as 1 throughout the example
        out, hidden = self.lstm(embed.view(len(x), 1, -1), hidden)
        self.lstm_out.append(out)
        self.lstm_hidden.append(hidden)
        output = self.hidden2tag(out.view(len(x), -1))
        return output

    def _init_hidden(self):
        # the dimension semantics are [num_layers, batch_size, hidden_size]
        return (torch.rand(1, 1, self.hidden_size),
                torch.rand(1, 1, self.hidden_size))



def run_lstm_worker(rank, size, epoch_limit=None, step_limit=None, federation_scheme='shuffle', queue=None, model_filepath=None):
    """
    Perform federated learning in a ring structure
    :param rank: index for specific data set
    :param size: total ring size
    :param queue: stores values and privacy accountant usage
    :param model_filepath: indicating where to save the model checkpoint
    :return:
    """

    # Every node gets same data for now
    EMBEDDING_SIZE = 6
    HIDDEN_SIZE = 6

    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

    idx_to_tag = ['DET', 'NN', 'V']
    tag_to_idx = {'DET': 0, 'NN': 1, 'V': 2}

    word_to_idx = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    def prepare_sequence(seq, to_idx):
        """Convert sentence/sequence to torch Tensors"""
        idxs = [to_idx[w] for w in seq]
        return torch.LongTensor(idxs)

    seq = training_data[0][0]
    inputs = prepare_sequence(seq, word_to_idx)

    model = LstmModule(EMBEDDING_SIZE, HIDDEN_SIZE, len(word_to_idx), len(tag_to_idx))
    criterion = nn.CrossEntropyLoss()

    accountant = PrivacyAccountant(model, step_epsilon=0.01)
    coordinator = ModelCoordinator(model, rank, size, federation_scheme)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epoch_limit):
        coordinator.recv()

        for sentence, tags in training_data:
            sentence = prepare_sequence(sentence, word_to_idx)
            target = prepare_sequence(tags, tag_to_idx)

            output = model(sentence)
            loss = criterion(output, target)
            loss.backward()

            # before
            accountant.privatize_grad()

            optimizer.step()
            optimizer.zero_grad()
        accountant.increment_epoch()

        inputs = prepare_sequence(training_data[0][0], word_to_idx)
        tag_scores = model(inputs)
        tag_scores = tag_scores.detach().numpy()
        tag = [idx_to_tag[idx] for idx in np.argmax(tag_scores, axis = 1)]
        printf(f"{rank: 4d} | {epoch: 5d} | {tag}     | {training_data[0][1]}", force=True)

        # Ensure that send() does not happen on the last epoch of the last node,
        # since this would send back to the first node (which is done) and hang
        if rank == size - 1 and epoch == epoch_limit - 1:
            # Only save if filename is given
            if model_filepath:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, model_filepath)
        else:
            coordinator.send()

    if queue:
        queue.put((tuple(datasets[rank].values()), accountant.compute_usage()))


if __name__ == "__main__":
    print("Rank | Epoch | Predicted | Actual")
    main(worker=run_lstm_worker)