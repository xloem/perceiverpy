import perceiver_pytorch as pp
import torch

from pytorch_model import PytorchModel

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

    # todo: this would work better if PytorchModel.update returned the individual losses

    def __init__(self, performer, databatches):
        self.performer = performer
        self.databatches = iter(databatches)
        self._nextidx = 0
        self.batch_losses = []

        #self.__iter = iter(databatches)
        #self.__iteridx = self._nextidx

    def train(self):
        self.batch_losses = [ SuperTrainer.BatchLoss(self), SuperTrainer.BatchLoss(self) ]
        self.batch_losses.sort()

        self.baseline = self.batch_losses[0].loss#2#0.5

        for batch_loss in self.batch_losses:
            yield batch_loss.idxs, batch_loss.idxs, batch_loss.losses

        # it can get stuff right by chance, and that happens with the first batch
        # so waiting for that first batch to bubble up fails
        # averaging it with something would help; i guess the hardest

        #hardest_loss = self.batch_losses[-1].loss
        #newest = self.batch_losses[0]
        newest = self.batch_losses[-1]
        hardest = newest
        epoch = 0
        while True:
            SuperTrainer.BatchLoss.sort(*self.batch_losses)
            #self.batch_losses.sort()
            #if self.batch_losses[-1] is newest:
            if self.batch_losses[-1].loss <= self.baseline:#(hardest_loss + newest.loss) / 2:
                # we learned a lot from our hardest example
                # we can set a mark and update what we know about all our examples
                epoch += 1
                easiest = self.batch_losses[0]

                # new data too
                #print('getting a new batch')
                try:
                    newest = self._newbatch()
                except StopIteration:
                    return
                newest.epoch = epoch
                #print('new data, loss =', newest.loss.item())
                yield hardest.idxs, newest.idxs, newest.losses

                max_loss = easiest if easiest > newest else newest

                for item in self.batch_losses:
                    item.epoch = epoch
                    # until OOPS is handled, below is commented out to prevent fitting on recent properties rather than new diversity
                    #if item is not hardest:
                    #    # OOPS: we'd probably only want to backpropagate if predictions are much worse
                    #    item.update()
                    if item.loss > max_loss:
                        max_loss = item
                    #elif item.loss < self.baseline:
                    #    self.baseline = (self.baseline + item.loss) / 2
                        #yield item

                hardest = max_loss
                self.baseline = (hardest.loss + easiest.loss) / 2 # this can hit 0.000, which likely confuses the model, overfitting it needleslly.  a minimum relating to the number of datapoints could make sense  
                                                                  # maybe instead of a loss baseline, it would make more sense to have predictions be correct.  [also option of predicting class and alternate class or such]
                    
                #yield max_loss.idxs, max_loss.loss
                #yield newest
            else:
                hardest = self.batch_losses[-1]
                hardest.update()
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
        if len(self.batch_losses) >= 256:
            if batch > self.batch_losses[0]:
                # this code of dropping the easiest batch, and placing the new batch at the head,
                # is kept to preserve the concept of the newest being the most interesting,
                # in case of redesign
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
                # .detach because gradients accumulate unbounded history when resliced during sorting
                batch.losses = losses[offset:offset+len(batch)].detach()
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

            shuf = torch.sort(losses).indices.cpu()

            offset = 0
            for batch in batches:
                idcs = shuf[offset:offset+len(batch)]
                if not torch.equal(idcs, torch.arange(offset, offset + len(batch))):
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

def test():
    batch_size = 4
    import torchvision
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import perceiver_pytorch as pp
    model = pp.Perceiver(input_channels=3, input_axis=2, num_freq_bands=6, max_freq=10.0, depth=6, num_latents=32, latent_dim=128, cross_heads=1, latent_heads=2, cross_dim_head=8, latent_dim_head=8, num_classes=10, attn_dropout=0.0, ff_dropout=0.0, weight_tie_layers=False)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    perform = PytorchModel(model, criterion, optimizer)

    trainloader = transformed_list(lambda batch: (batch[0].permute(0,2,3,1), batch[1]), trainloader)
    testloader = transformed_list(lambda batch: (batch[0].permute(0,2,3,1), batch[1]), testloader)


    import time
    last_time = time.time()
    start_time = last_time
    last_avg_loss = None
    running_loss = 0.0
    running_baseline = 0.0
    running_ct = 0
    idx = 0
    last_idxs = None
    trainer = SuperTrainer(perform, trainloader)
    for learned_idxs, idxs, losses in trainer.train():
        running_loss += torch.sum(losses)
        running_baseline += trainer.baseline * len(losses)
        running_ct += len(losses)
        idx += len(losses)

        if last_idxs is None:
            last_idxs = learned_idxs
        elif not torch.equal(last_idxs, learned_idxs):
            print('%s maxloss=%.3f avg_loss=%.3f newbaseline=%.3f' % (str([i.item() for i in last_idxs]), torch.max(losses), running_loss / running_ct, trainer.baseline))
            last_idxs = learned_idxs
        if idx % 256 == 0 and running_ct > len(losses):
            cur_time = time.time()
            if cur_time - last_time > 0.2:
                avg_loss = running_loss / running_ct
                avg_baseline = running_baseline / running_ct
                if last_avg_loss is None:
                    last_avg_loss = float('nan')
                lossrate = (last_avg_loss - avg_loss) / (cur_time - last_time)
                if not lossrate < 0 or cur_time - last_time >= 60:
                    print('%5d avg_loss=%.3f %.5f loss/min avg_baseline=%.3f' % (idx, avg_loss, 60*lossrate, avg_baseline))
                    last_avg_loss = avg_loss
                    last_time = cur_time
                    running_loss = 0
                    running_baseline = 0
                    running_ct = 0

    print('trained')
    import numpy as np
    tests, labels = next(iter(testloader))
    def preds2words(preds):
        return preds.detach().cpu().numpy()
    print('labels:', [classes[l] for l in labels])
    preds = perform.predict(*tests)#.permute(0,2,3,1))
    preds = preds.detach().cpu().numpy()
    labels = [np.argmax(i) for i in preds]
    print('preds:', [classes[l] for l in labels])

#perform = PytorchModel(pp.Perceiver(input_channels=1, input_axis=1, depth=6, fourier_encode_data=False, num_freq_bands=None, max_freq=None))
if __name__ == '__main__':
    test()

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
