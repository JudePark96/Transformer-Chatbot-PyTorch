"""
Author: Jude Park <judepark@kookmin.ac.kr>
"""


from config.base_experiment_config import get_base_config, get_device_setting
from utils.build_vocab import Vocabulary
from torch.utils.data import DataLoader
from utils.data_utils import load_data
from utils.dataloder import get_loader
from torch.nn import CrossEntropyLoss
from model.transformer import Net
from torch.nn import Module
from tqdm import tqdm


import torch.optim as optim
import torch as T


class Trainer(object):

    def __init__(self,
                 args: dict,
                 vocab: Vocabulary,
                 model: Module,
                 optimizer: optim,
                 output_path: str) -> None:
        super().__init__()

        if vocab.get_vocab_size() != args['vocab_size']:
            raise ValueError('vocabulary has not been initiated.')

        self.args = args
        self.vocab = vocab
        self.model = model.to(get_device_setting())
        self.optimizer = optimizer
        self.output_path = output_path
        self.train_loss = []

        # define what loss function is.
        self.loss_fn = CrossEntropyLoss(ignore_index=self.args['pad_idx']).to(get_device_setting())

    def train(self, epoch: int, train_loader: DataLoader, valid_loader: DataLoader) -> None:

        # set the model to train mode.
        self.model.train()

        for ep_iter in tqdm(range(1, epoch + 1)):
            print(f'********** epoch number: {ep_iter} **********')
            for i, (question, answer) in tqdm(enumerate(train_loader)):
                self.optimizer.zero_grad()
                output = self.model(question, answer)
                print(output.shape)
                output = output.view(-1, output.size(-1))
                answer = answer.view(-1).long()
                loss = self.loss_fn(output, answer.to(get_device_setting()))
                print(f'********** training loss: {loss.item()} **********')
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0 and epoch > 2:
                    self.evaluate(i, valid_loader)
                    T.save(self.model.state_dict(), self.output_path + f'model-{ep_iter}-{i}.pt')

    def evaluate(self, iteration: int, valid_loader: DataLoader):
        def decode_sequences(question, prediction, answer):
            question_ids = question.tolist()
            pred_ids = prediction.max(dim=-1)[1].tolist()
            answer_ids = answer.tolist()

            decoded_question = []
            decoded_prediction = []
            decoded_answer = []

            for questions in question_ids:
                seq = ' '.join([self.vocab.idx2token[question_id] for question_id in questions])
                decoded_question.append(seq)

            for preds in pred_ids:
                seq = ' '.join([self.vocab.idx2token[pred_id] for pred_id in preds])
                decoded_prediction.append(seq)

            for answers in answer_ids:
                seq = ' '.join([self.vocab.idx2token[answer_id] for answer_id in answers])
                decoded_answer.append(seq)

            for q, p, a in zip(decoded_question, decoded_prediction, decoded_answer):
                print('********** decoded result **********')
                print(q + '\n')
                print(p + '\n')
                print(a + '\n')


        print(f'********** evaluating start **********')

        self.model.eval()

        for i, (question, answer) in tqdm(enumerate(valid_loader)):
            output = self.model(question, answer)
            decode_sequences(question, output, answer)

            output = output.view(-1, output.size(-1))
            answer = answer.view(-1).long()

            loss = self.loss_fn(output, answer.to(get_device_setting()))
            print(f'********** evaluating loss: {loss.item()} **********')


if __name__ == '__main__':
    train, valid, train_y, valid_y, corpus = load_data('./rsc/data/chatbot_korean.csv')
    vocab = Vocabulary(corpus)
    vocab.build_vocab()
    loader = get_loader(train, train_y, vocab, 64, 32, True)
    args = get_base_config()

    model = Net(args)
    optimizer = optim.Adam(params=model.parameters(), lr=args['lr'])
    Trainer(args, vocab, model, optimizer).train(1, loader)