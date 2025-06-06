import numpy as np
import torch

import openood.utils.comm as comm
from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger

import matplotlib.pylab as pyb
import matplotlib.pyplot as plt


class TrainOEPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        train_oe_loader = loader_dict['oe']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)
        
        # print(net.fc.weight.cpu().detach().numpy().shape)  # (10, 512)
        # print(net.fc.bias.cpu().detach().numpy().shape)   # (10,)
        # nets = [[np.mean(net.fc.weight.cpu().detach().numpy()[0]), net.fc.bias.cpu().detach().numpy()[0]]]
        # losses = []
        # init trainer and evaluator
        trainer = get_trainer(net, [train_loader, train_oe_loader], None,
                              self.config)
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            # nets.append([np.mean(net.fc.weight.cpu().detach().numpy()[0]), net.fc.bias.cpu().detach().numpy()[0]])
            # losses.append(train_metrics['loss'])
            val_metrics = evaluator.eval_acc(net, val_loader, None, epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)
                
        # plot param changing dynamics
        # cm = pyb.get_cmap('terrain')
        # fig, ax = plt.subplots()
        # plt.xlabel('Bias of Class-1')
        # plt.ylabel('Mean Weight of Class-1')
        # i = ax.imshow(losses, cmap=cm, interpolation='nearest'
        #               ,extent=[-10, 10, -10, 10]
        #               )
        
        # net_weights, net_biases = zip(*nets)
        # ax.scatter(net_biases, net_weights, c='r', marker='->')
        # ax.plot(net_biases, net_weights, c='r')

        # # fig.colorbar(i)
        # plt.savefig('./oe-training-dynamics-e0.png', dpi=700)
        

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc(net, test_loader)

        if comm.is_main_process():
            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)
