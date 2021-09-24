#from torch.utils.tensorboard import SummaryWriter
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint


class BertSimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.classifier_optimizer = kwargs['classifier_optimizer']
        self.scheduler = kwargs['scheduler']
        self.C = self.args.C
        #self.writer = SummaryWriter()
        self.logdir = "200runs"
        logging.basicConfig(filename=os.path.join(self.logdir,'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        if self.args.eval:
            self.linear_classifier = kwargs['classifier_model'].to(self.args.device)
            self.linear_criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)


        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def forward(self, images1, images2, caption_encodings):

        images1 = images1.to(self.args.device)
        images2 = images2.to(self.args.device)
        caption_encodings = caption_encodings.to(self.args.device)

        with autocast(enabled=self.args.fp16_precision):
            visual_embedding1, visual_embedding2, sentence_embedding = self.model(images1, images2, caption_encodings)
            logits, labels = self.info_nce_loss(torch.cat((visual_embedding1, visual_embedding2), dim=0))
            loss = self.criterion(logits, labels)
            logits, labels = self.info_nce_loss(torch.cat((visual_embedding1, sentence_embedding), dim=0))
            loss += self.C*self.criterion(logits, labels)
            logits, labels = self.info_nce_loss(torch.cat((visual_embedding2, sentence_embedding), dim=0))
            loss += self.C*self.criterion(logits, labels)

        return loss

    def train(self, loaders):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        #save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train(True)
                else:
                    self.model.train(False)
                
                running_loss = 0.0
                #running_1acc = 0.0
                #running_5acc = 0.0

                for images1, images2, caption_encoding in tqdm(loaders[phase]):
                    if phase == "train":
                        loss = self.forward(images1, images2, caption_encoding)

                        self.optimizer.zero_grad()

                        scaler.scale(loss).backward()

                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        with torch.no_grad():
                            loss = self.forward(images1, images2, caption_encoding)

                    #top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    running_loss += loss
                    #running_1acc += top1[0]
                    #running_5acc += top5[0]

                loss = running_loss / len(loaders[phase])
                #acc1 = running_1acc / len(loaders[phase])
                #acc5 = running_5acc / len(loaders[phase])

                #self.writer.add_scalar('loss/' + phase, loss, global_step=epoch_counter)
                #self.writer.add_scalar('top1/' + phase, acc1, global_step=epoch_counter)
                #self.writer.add_scalar('top5/' + phase, acc5, global_step=epoch_counter)
                logging.debug(f"Epoch: {epoch_counter} loss/{phase} {loss}")
                #logging.debug(f"Epoch: {epoch_counter} top1/{phase} {acc1}")
                #logging.debug(f"Epoch: {epoch_counter} top5/{phase} {acc5}")

                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.logdir, checkpoint_name))

        logging.info("Training has finished.")
        # save model checkpoints
        logging.info(f"Model checkpoint and metadata has been saved at {self.log_dir}.")

    def classifier_forward(self, x, labels):
        embedding = self.model.visual_backbone(x)
        logits = self.linear_classifier(embedding)
        loss = self.linear_criterion(logits, labels).to(self.args.device)

        return loss, logits

    def train_linear_classifier(self, num_epochs, loaders):
        for epoch in range(num_epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train(True)
                else:
                    self.model.train(False)
                
                running_loss = 0.0
                running_1acc = 0.0
                running_5acc = 0.0
                for images, labels in tqdm(loaders[phase]):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    if phase == "train":
                        loss, logits = self.classifier_forward(images, labels)

                        self.classifier_optimizer.zero_grad()

                        loss.backward()

                        self.classifier_optimizer.step()
                    else:
                        with torch.no_grad():
                            loss, logits = self.classifier_forward(images, labels)
                        
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    running_loss += loss
                    running_1acc += top1[0]
                    running_5acc += top5[0]

                loss = running_loss / len(loaders[phase])
                acc1 = running_1acc / len(loaders[phase])
                acc5 = running_5acc / len(loaders[phase])

                logging.debug(f"Epoch: {epoch_counter} classifier_loss/{phase} {loss}")
                logging.debug(f"Epoch: {epoch_counter} top1/{phase} {acc1}")
                logging.debug(f"Epoch: {epoch_counter} top5/{phase} {acc5}")

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        if self.args.eval:
            self.linear_classifier = kwargs['classifier_model'].to(self.args.device)
            self.classifier_optimizer = kwargs['classifier_optimizer']
            self.linear_criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def classifier_forward(self, x, labels):
        embedding = self.model.backbone(x)
        logits = self.linear_classifier(embedding)
        loss = self.linear_criterion(logits, labels).to(self.args.device)

        return loss, logits

    def train_linear_classifier(self, num_epochs, loaders):
        for epoch in range(num_epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train(True)
                else:
                    self.model.train(False)
                
                running_loss = 0.0
                running_1acc = 0.0
                running_5acc = 0.0
                for images, labels in tqdm(loaders[phase]):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    if phase == "train":
                        loss, logits = self.classifier_forward(images, labels)

                        self.classifier_optimizer.zero_grad()

                        loss.backward()

                        self.classifier_optimizer.step()
                    else:
                        with torch.no_grad():
                            loss, logits = self.classifier_forward(images, labels)
                        
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    running_loss += loss
                    running_1acc += top1[0]
                    running_5acc += top5[0]

                loss = running_loss / len(loaders[phase])
                acc1 = running_1acc / len(loaders[phase])
                acc5 = running_5acc / len(loaders[phase])

                self.writer.add_scalar('classifier_loss/' + phase, loss, global_step=epoch)
                self.writer.add_scalar('classifier_top1/' + phase, acc1, global_step=epoch)
                self.writer.add_scalar('classifier_top5/' + phase, acc5, global_step=epoch)

    def train(self, loaders):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train(True)
                else:
                    self.model.train(False)


                
                running_loss = 0.0
                running_1acc = 0.0
                running_5acc = 0.0

                for images, _ in tqdm(loaders[phase]):
                    if phase == "train":

                        images = torch.cat(images, dim=0)

                        images = images.to(self.args.device)

                        with autocast(enabled=self.args.fp16_precision):
                            features = self.model(images)
                            logits, labels = self.info_nce_loss(features)
                            loss = self.criterion(logits, labels)

                        self.optimizer.zero_grad()

                        scaler.scale(loss).backward()

                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        with torch.no_grad():
                            images = torch.cat(images, dim=0)

                            images = images.to(self.args.device)

                            with autocast(enabled=self.args.fp16_precision):
                                features = self.model(images)
                                logits, labels = self.info_nce_loss(features)
                                loss = self.criterion(logits, labels)

                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    running_loss += loss
                    running_1acc += top1[0]
                    running_5acc += top5[0]

                    del features, logits, labels

                loss = running_loss / len(loaders[phase])
                acc1 = running_1acc / len(loaders[phase])
                acc5 = running_5acc / len(loaders[phase])

                self.writer.add_scalar('loss/' + phase, loss, global_step=epoch_counter)
                self.writer.add_scalar('top1/' + phase, acc1, global_step=epoch_counter)
                self.writer.add_scalar('top5/' + phase, acc5, global_step=epoch_counter)

                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        logging.info("Training has finished.")
        # save model checkpoints
