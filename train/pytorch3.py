import perceiver_pytorch as pp
import torch

import sys

class Perform:
    def __init__(self, model, loss_fn = torch.nn.CrossEntropyLoss(reduction='none'), optimizer = None, dev = 'cuda:0'):
        if optimizer is None:
            optimizer = torch.optim.Rprop()
            #optimizer = 1e-3
        if type(optimizer) is float:
            #optimizer = torch.optim.Adam(model.parameters(), lr=optimizer)
            optimizer = torch.optim.SGD(model.parameters(), lr=optimizer)
        if type(dev) is str:
            dev = torch.device(dev)
        model.to(dev)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_state = optimizer.state_dict()
        self.dev = dev
        self.last_result = None
    @staticmethod
    def _tensor(data, dev):
        if type(data) is not torch.Tensor:
            data = torch.tensor(data, device=dev)
        return data
    def predict(self, *data):
        return self.predict_many(data)
    def predict_many(self, *datas):
        self.optimizer.zero_grad() # reset gradients of parameters
        self.last_predictions = []

        for data in datas:

            data = torch.stack([self._tensor(item, None) for item in data])
            data = data.to(self.dev)

            self.last_predictions.append(self.model(data))

        self.last_predictions = torch.cat(self.last_predictions)
        return self.last_predictions

    def update(self, *better_results):
        better_results = torch.stack([self._tensor(result, None) for result in better_results])
        better_results = better_results.to(self.dev)

        losses = self.loss_fn(self.last_predictions, better_results)
        loss = torch.mean(losses)

        loss.backward() # backpropagate prediction loss, deposit gradients of loss wrt each parameter
        self.optimizer.step() # adjust parameters by gradients collected in backward()
        return losses

# i think sklearn has arch around this, unsure
      #  num_freq_bands: Number of freq bands, with original value (2 * K + 1)
      #  depth: Depth of net.
      #  max_freq: Maximum frequency, hyperparameter depending on how
      #      fine the data is.
      #  freq_base: Base for the frequency
      #  input_channels: Number of channels for each token of the input.
      #  input_axis: Number of axes for input data (2 for images, 3 for video)
      #  num_latents: Number of latents, or induced set points, or centroids.
      #      Different papers giving it different names.
      #  latent_dim: Latent dimension.
      #  cross_heads: Number of heads for cross attention. Paper said 1.
      #  latent_heads: Number of heads for latent self attention, 8.
      #  cross_dim_head: Number of dimensions per cross attention head.
      #  latent_dim_head: Number of dimensions per latent self attention head.
      #  num_classes: Output number of classes.
      #  attn_dropout: Attention dropout
      #  ff_dropout: Feedforward dropout
      #  weight_tie_layers: Whether to weight tie layers (optional).
      #  fourier_encode_data: Whether to auto-fourier encode the data, using
      #      the input_axis given. defaults to True, but can be turned off
      #      if you are fourier encoding the data yourself.
      #  self_per_cross_attn: Number of self attention blocks per cross attn.

import functools

# very simple relation possible here
# of learning rate, vs reduction in loss

class SuperTrainer:
    # license: gpl-3
    # focus on the area with the greatest loss

    # todo: this would work better if Performer.update returned the individual losses
    #    then disparate data could be collected in the same batch which would greatly speed learning important differences

    # human supervision: we would want to queue and rank data based on how extreme it is, and how lacking the human-label is in the known data.  this could provide for humans labeling mnost effectivelly.

    def __init__(self, performer, databatches, num_batches=256):
        self.performer = performer
        self.databatches = iter(databatches)
        self._nextidx = 0
        self.num_batches = num_batches
        self.batch_losses = []

        #self.__iter = iter(databatches)
        #self.__iteridx = self._nextidx

    def train(self):
        self.batch_losses = [ SuperTrainer.BatchLoss(self), SuperTrainer.BatchLoss(self) ]
        self.batch_losses.sort()

        self.baseline = float(self.batch_losses[0])#2#0.5
        newest = self.batch_losses[-1]
        hardest = newest
        epoch = 0
        while True:
            SuperTrainer.BatchLoss.sort(*self.batch_losses)
            #self.batch_losses.sort()
            #for batch_loss in self.batch_losses:
            #    print(str(batch_loss))

            #if hardest is not self.batch_losses[-1]:
            #    # mix contents of hard batches together, to give easy parts a chance to leave, and disparate parts a chance to share training
            #    hardest.shuffle(self.batch_losses[-1])
            #    hardest.loss = self.batch_losses[-1].loss = (hardest.loss + self.batch_losses[-1].loss) / 2

            if self.batch_losses[-1] <= self.baseline:#(hardest_loss + newest.loss) / 2:
                # we learned a lot from our hardest example
                # we can set a mark and update what we know about all our examples
                epoch += 1

                easiest = self.batch_losses[0]

                # new data too
                print('getting new batches')
                while True:
                    newest = self._newbatch()
                    newest.epoch = epoch
                    #print('new data, loss =', newest.loss.item())
                    if newest > self.baseline:
                        break
                    else:
                        self.baseline = (self.baseline + float(newest)) / 2

                #easiest.update()

                print('going over old items')

                max_loss = easiest if easiest > newest else newest

                if max_loss > self.baseline:

                    min_loss = newest
                    #SuperTrainer.BatchLoss.shuffle(*self.batch_losses[:self.num_batches])
    
                    for item in self.batch_losses:
                        item.epoch = epoch
                        if item not in last_updated and item is not newest:
                            # OOPS: we'd probably only want to backpropagate if predictions are much worse
                            item.update()
                        if item > max_loss:
                            max_loss = item
                        if item < min_loss:
                            min_loss = item
                        #elif item.loss < self.baseline:
                        #    self.baseline = (self.baseline + item.loss) / 2
                            #yield item

                    hardest = max_loss
                    easiest = min_loss
                    self.baseline = float(easiest) + (float(hardest) - float(easiest)) / 3 # this can hit 0.000, which likely confuses the model, overfitting it needleslly.  a minimum relating to the number of datapoints could make sense  
                                                                  # maybe instead of a loss baseline, it would make more sense to have predictions be correct.  [also option of predicting class and alternate class or such]
                if self.baseline < 0.25:
                    self.baseline = 0.25
                    
                yield max_loss.idxs, float(max_loss)#.losses
            
                print('working on hardest items, target:', self.baseline)
                self.trainer.optimizer.load_state_dict(self.trainer.optimizer_state)
                #yield newest
            else:
                sys.stderr.write(str(self.batch_losses[-1]) + '\r')
                last_updated = self.batch_losses[-1:]#[-2:]
                #SuperTrainer.BatchLoss.shuffle(*last_updated)
                for batch in last_updated:
                    batch.update()
                #yield hardest


            # # study hardest example
            # batch_losses[-1].update()
            # yield batch_losses[-1]

            # if batch_losses[-1] is not interest:
            #     # a new example is hardest, meaning we have learned about our hardest example
            #     print('studied', interest, ' hardest =', batch_losses[-1], ' easiest =', batch_losses[0])

            #     # update easiest example with new knowledge
            #     batch_losses[0].update()

            #     if batch_losses[0] <= batch_losses[1]:
            #         # new knowledge didn't change easiness of easiest example
            #         # get more diverse data
            #         interest = self._newbatch()
            #     else:
            #         # new knowledge mutated results from old knowledge
            #         yield batch_losses[0]

    def _newbatch(self):
        batch = SuperTrainer.BatchLoss(self)
        if len(self.batch_losses) >= self.num_batches:
            self.batch_losses = self.batch_losses[:self.num_batches]
            self.batch_losses[:-1] = self.batch_losses[1:]
            self.batch_losses[-1] = batch
        else:
            self.batch_losses.append(batch)
        return batch

    @functools.total_ordering
    class BatchLoss:
        def __init__(self, trainer):
            self.trainer = trainer
            self.batch = next(self.trainer.databatches)
            idx = trainer._nextidx
            trainer._nextidx += len(self)
            self.idxs = torch.arange(idx, trainer._nextidx)
            self.update()
        def update(*batches):
            inputslabels = [batch.batch for batch in batches]
            inputs = [inputs for inputs, labels in inputslabels]
            labels = torch.cat([labels for inputs, labels in inputslabels])
            self = batches[0]
            self.trainer.performer.predict_many(*inputs)
            losses = self.trainer.performer.update(*labels)
            #loss = torch.mean(losses)
            offset = 0
            for batch in batches:
                batch.losses = losses[offset:offset+len(batch)]
                batch.loss = torch.max(batch.losses).item()
                offset += len(batch)
                #batch.loss = torch.mean(batch.losses)
            return losses
        def shuffle(*batches):
            idxs = torch.cat([batch.idxs for batch in batches])
            inputs = torch.cat([batch.batch[0] for batch in batches])
            labels = torch.cat([batch.batch[1] for batch in batches])
            losses = torch.cat([batch.losses for batch in batches])
            shuf = torch.randperm(len(idxs))
            offset = 0
            for batch in batches:
                idcs = shuf[offset:offset+len(batch)]
                batch.idxs = idxs[idcs]
                batch.batch = (inputs[idcs], labels[idcs])
                batch.losses = losses[idcs]
                batch.loss = torch.max(batch.losses).item()
                offset += len(batch)
        def sort(*batches):
            idxs = torch.cat([batch.idxs for batch in batches])
            inputs = torch.cat([batch.batch[0] for batch in batches])
            labels = torch.cat([batch.batch[1] for batch in batches])
            losses = torch.cat([batch.losses for batch in batches])

            shuf = torch.sort(losses).indices

            offset = 0
            for batch in batches:
                idcs = shuf[offset:offset+len(batch)]
                batch.idxs = idxs[idcs]
                batch.batch = (inputs[idcs], labels[idcs])
                batch.losses = losses[idcs]
                batch.loss = torch.max(batch.losses).item()
                offset += len(batch)
        def __len__(self):
            return len(self.batch[0])
        def __float__(self):
            return self.loss
        def __eq__(self, other):
            return float(self) == float(other)
        def __lt__(self, other):
            return float(self) < float(other)
        def __str__(self):
            return str(float(self)) + ' ' + str([*zip([float(idx) for idx in self.idxs], [float(loss) for loss in self.losses])])
            

class transformed_list:
    def __init__(self, mutation, src = []):
        self.mutation = mutation
        self.src = src
    def __getitem__(self, index):
        if type(index) is tuple:
            return transformed_list(self.mutation, self.src[index])
        else:
            return self.mutation(self.src[index])
    def __iter__(self):
        return (self.mutation(item) for item in self.src)
    def __len__(self):
        return len(self.src)

class Test:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        import torchvision
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        import perceiver_pytorch as pp
        self.model = pp.Perceiver(input_channels=3, input_axis=2, num_freq_bands=6, max_freq=10.0, depth=6, num_latents=32, latent_dim=128, cross_heads=1, latent_heads=2, cross_dim_head=8, latent_dim_head=8, num_classes=10, attn_dropout=0.0, ff_dropout=0.0, weight_tie_layers=False)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.perform = Perform(self.model, self.criterion, self.optimizer, dev = 'cuda:0')

        self.trainloader = transformed_list(lambda batch: (batch[0].permute(0,2,3,1), batch[1]), self.trainloader)
        self.testloader = transformed_list(lambda batch: (batch[0].permute(0,2,3,1), batch[1]), self.testloader)


        self.trainer = SuperTrainer(self.perform, self.trainloader)
    def test(self):
        import time
        last_time = time.time()
        starting_loss = None
        #running_loss = 0.0
        #running_last = -1
        idx = 0
        for i, loss in self.trainer.train():
            if starting_loss is None:
                starting_loss = loss
            #running_loss += loss.item()
            #if idx % 16 == 0:
            #    cur_time = time.time()
            #    if cur_time - last_time > 0.2:
            #        last_time = cur_time
            #        print('%5d loss=%.3f avg=%.3f' % (i, loss, running_loss / (idx - running_last)))
            #        running_loss = 0
            #        running_last = idx
            print('%s maxloss=%.3f newbaseline=%.3f %.3f loss/s' % ([j.item() for j in i], loss, self.trainer.baseline, (starting_loss - loss) / (time.time() - last_time)))
            idx += 1
    
        print('trained')
        import numpy as np
        tests, labels = next(iter(self.testloader))
        def preds2words(preds):
            return preds.detach().cpu().numpy()
        print('labels:', [self.classes[l] for l in labels])
        preds = self.perform.predict(*tests)#.permute(0,2,3,1))
        preds = preds.detach().cpu().numpy()
        labels = [np.argmax(i) for i in preds]
        print('preds:', [self.classes[l] for l in labels])

#perform = Perform(pp.Perceiver(input_channels=1, input_axis=1, depth=6, fourier_encode_data=False, num_freq_bands=None, max_freq=None))
if __name__ == '__main__':
    Test().test()

            # for datasets, it would be pretty useful to group data based on similar lossiness
            #   then we wouldn't have to reuse old data to have known hard properties
            #       hobbyist-algorithm, wonder what research name is:
            #            1 randomly sample data
            #            2 differentiate random sample from group
            #            3 sample data based on similarity to random sample
            #            4 repeat 2&3 to produce a group of data that has a particular property
            #            5 present group to user or classifier, separate from sample that lacks property.  produce name for property.
            #                - the property defines a data split.
            #            6 repeat 1-5, using known data splits to control evenly for known variables, to enumerate properties.
            #            7 repeat all properties, using known data splits to control evenly for known variables, to deweight random order of process
            #        #2 is the core part, that makes the grouping approach useful in diverse domains.
