import perceiver_pytorch as pp
import torch

class Perform:
    def __init__(self, model, loss_fn = torch.nn.CrossEntropyLoss(reduction='none'), optimizer = None, dev = 'cuda:0'):
        if optimizer is None:
            #optimizer = torch.optim.Rprop()
            optimizer = 1e-3
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


def test():
    import torchvision
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import perceiver_pytorch as pp
    model = pp.Perceiver(input_channels=3, input_axis=2, num_freq_bands=6, max_freq=10.0, depth=6, num_latents=32, latent_dim=128, cross_heads=1, latent_heads=2, cross_dim_head=8, latent_dim_head=8, num_classes=10, attn_dropout=0.0, ff_dropout=0.0, weight_tie_layers=False)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    perform = Perform(model, criterion, optimizer, dev = 'cuda:0')

    import time
    last_time = time.time()
    start_time = last_time
    starting_loss = None
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        perform.predict(*inputs.permute(0,2,3,1))  
        losses = perform.update(*labels)
        maxloss = torch.max(losses)
        if starting_loss is None or maxloss > starting_loss:
            starting_loss = maxloss
        running_loss += torch.sum(losses)
        if i % 16 == 0:
            cur_time = time.time()
            if cur_time - last_time > 0.2:
                last_time = cur_time
                lossrate = (starting_loss - maxloss) / (cur_time - start_time)
                print('%5d maxloss=%.3f running_loss=%.3f %.5f loss/s' % (i, maxloss, running_loss, lossrate))

    print('trained')
    import numpy as np
    tests, labels = iter(testloader).next()
    def preds2words(preds):
        return preds.detach().cpu().numpy()
    print('labels:', [classes[l] for l in labels])
    preds = perform.predict(*tests.permute(0,2,3,1))
    preds = preds.detach().cpu().numpy()
    labels = [np.argmax(i) for i in preds]
    print('preds:', [classes[l] for l in labels])

#perform = Perform(pp.Perceiver(input_channels=1, input_axis=1, depth=6, fourier_encode_data=False, num_freq_bands=None, max_freq=None))
if __name__ == '__main__':
    test()
