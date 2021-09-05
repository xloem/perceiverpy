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


def test():
    import torchvision
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import perceiver_pytorch as pp
    model = pp.Perceiver(input_channels=3, input_axis=2, fourier_encode_data=True, num_freq_bands=6, max_freq=10.0, depth=6, num_latents=32, latent_dim=128, cross_heads=1, latent_heads=2, cross_dim_head=8, latent_dim_head=8, num_classes=10, attn_dropout=0.0, ff_dropout=0.0, weight_tie_layers=False)
    #print(next(model.parameters()), 'sometimes i see model parameters that start for some reason all near zero.  that is a bug.  parameters above should be near unit magnitude.')
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    perform = PytorchModel(model, criterion, optimizer, dev = 'cuda:0')

    import time
    last_time = time.time()
    start_time = last_time
    start_loss = None
    last_avg_loss = None
    running_loss = 0.0
    running_ct = 0
    for epoch in range(4):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            perform.predict(*inputs.permute(0,2,3,1))  
            losses = perform.update(*labels)
            maxloss = torch.max(losses)
            if start_loss is None or maxloss > start_loss:
                start_loss = maxloss
            running_loss += torch.sum(losses)
            running_ct += len(losses)
            if (i*len(losses)) % 1024 == 0 and running_ct > len(losses):
                cur_time = time.time()
                if cur_time - last_time > 0.2:
                    avg_loss = running_loss / running_ct
                    if last_avg_loss is None:
                        last_avg_loss = float('nan')
                    lossrate = (last_avg_loss - avg_loss) / (cur_time - last_time)
                    if not lossrate < 0 or cur_time - last_time >= 60:
                        print('%d %5d avg_loss=%.3f %.5f loss/min' % (epoch+1, i * len(losses), avg_loss, 60*lossrate))
                        last_avg_loss = avg_loss
                        last_time = cur_time
                        running_loss = 0
                        running_ct = 0

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
