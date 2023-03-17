import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import models
from retinanet.dataloader import CelebA_Dataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer, Tensor
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--celeb_path', help='Path to COCO directory')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    dataset_train = CelebA_Dataset(transform = transforms.Compose([Resizer(), Tensor()]))
    dataloader_train = DataLoader(dataset_train, num_workers=3, batch_size=32)

    model = models.BackBone(num_attr=40)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.training = True

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = []
    reg_loss = []
    attr_loss = []


    model.train()
    model.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))


    for epoch_num in range(parser.epochs):

        model.train()
        model.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    attribute_loss, regression_loss = model([data['img'].cuda().float(), data['annot']])
                else:
                    attribute_loss, regression_loss = model([data['img'].float(), data['annot']])

                loss = attribute_loss + regression_loss

                reg_loss.append(regression_loss.detach().cpu().numpy())
                attr_loss.append(attribute_loss.detach().cpu().numpy())
                loss_hist.append(loss.detach().cpu().numpy())
                epoch_loss.append(loss.detach().cpu().numpy())
                # Log losses for visulization purpose
                np.savez('files.npz', x=reg_loss, y=attr_loss, z=loss_hist)

                #   Visualize on tensorboard
                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                       epoch_num, iter_num, float(attribute_loss), float(regression_loss), np.mean(loss_hist)))

                del attribute_loss
                del regression_loss

            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))
        torch.save(model.module, f'./checkpoints/CelebA_model{epoch_num}.pt')

    torch.save(model, 'model_final.pt')


if __name__ == '__main__':
    main()
