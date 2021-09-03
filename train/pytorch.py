import perceiver_pytorch as pp
import torch

class Perform:
    def __init__(self, model, loss_fn = torch.nn.CrossEntropyLoss(), optimizer = None):
        if optimizer is None:
            optimizer = 1e-3
        if type(optimizer) is float:
            optimizer = torch.optim.SGD(model.parameters(), lr=optimizer)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.last_result = None
    @staticmethod
    def _tensor(data):
        if type(data) is not torch.Tensor:
            data = torch.Tensor(data)
        return data
    def predict(self, *data):
        # data would be tensors
        self.last_predictions = self.model(torch.Tensor(data))
        return self.last_predictions
    def update(self, *better_results):
        # be nice if this associated with the gradients calculated for data, if they are used, probably pulloutable
            # might need to put these lists on device
        loss = self.loss_fn(self.last_predictions, torch.Tensor(better_results))

        self.optimizer.zero_grad() # reset gradients of parameters
        loss.backward() # backpropagate prediction loss, deposit grdaients of loss wrt each parameter
        self.optimizer.step() # adjust parameters by gradients collected in backward()
        return loss

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

perform = Perform(pp.Perceiver(input_channels=1, input_axis=1, depth=6, fourier_encode_data=False, num_freq_bands=None, max_freq=None))
