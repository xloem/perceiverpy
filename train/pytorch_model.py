import torch

import os
import weakref

from util import name2path, path2name, stack2name, dict2name

_models = weakref.WeakValueDictionary()

default_device = torch.device(('cpu','cuda:0')[torch.cuda.is_available()])

def NamedModel(model_class, dev = default_device, **kwparams):
    name = model_class.__name__ + '-' + dict2name(kwparams)
    model = _models.get(name)
    if model is None:
        model = _models.setdefault(name, model_class(**kwparams))
        model.name = name
        if type(dev) is str:
            dev = torch.device(dev)
        model.to(dev)
        model.device = dev
    return model

class PytorchModel:
    # license: gpl-3
    def __init__(self, namedmodel, loss_fn = torch.nn.CrossEntropyLoss(reduction='none'), optimizer = None, state_filename = None):
        if optimizer is None:
            #optimizer = torch.optim.Rprop()
            optimizer = 1e-3
        if type(optimizer) is float:
            #optimizer = torch.optim.Adam(model.parameters(), lr=optimizer)
            optimizer = torch.optim.SGD(model.parameters(), lr=optimizer)
        model = namedmodel
        if hasattr(model, 'name') and hasattr(model, 'device'):
            modelname = model.name
            self.dev = model.device
        else:
            modelname = model.__class__.__name__
            # could we get the dev off the model
            self.dev = default_device
            model.to(self.dev)
        self.model = model
        if state_filename is None:
            state_filename = name2path(modelname, stack2name())
        self.state_filename = state_filename
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_state = optimizer.state_dict()
        self.last_result = None
        if os.path.exists(state_filename):
            self.load()
    def save(self):
        os.makedirs(os.path.dirname(self.state_filename), exist_ok=True)
        torch.save(self.model.state_dict(), self.state_filename)
    def load(self):
        self.model.load_state_dict(torch.load(self.state_filename))
    def to_snapshot(self):
        return (self.optimizer.state_dict(), self.model.state_dict())
    def from_snapshot(self, snapshot):
        optimizer_state, model_state = snapshot
        self.optimizer.load_state_dict(optimizer_state)
        self.model.load_state_dict(model_state)
    @staticmethod
    def _tensor(data, dev):
        if type(data) is not torch.Tensor:
            data = torch.tensor(data, device=dev)
        return data
    def predict(self, *data):
        return self.predict_many(data)
    def predict_many(self, *datas):
        # todo: this method, or a method like it, could use its call stack to select models from a repository automatically.

        self.optimizer.zero_grad() # reset gradients of parameters
        self.last_predictions = []

        for data in datas:

            data = torch.stack([self._tensor(item, None) for item in data])
            data = data.to(self.dev)

            self.last_predictions.append(self.model(data))

        self.last_predictions = torch.cat(self.last_predictions)
        return self.last_predictions
    def losses(self, *better_results):
        better_results = torch.stack([self._tensor(result, None) for result in better_results])
        better_results = better_results.to(self.dev)
        return self.loss_fn(self.last_predictions, better_results)
    def predict_losses(self, *data, results):
        with torch.no_grad():
            self.predict(*data)
            return self.losses(*results)
    def predict_many_losses(self, *datas, results):
        with torch.no_grad():
            self.predict_many(*datas)
            return self.losses(*results)
    def update(self, *better_results, losses=None):
        if losses is None:
            losses = self.losses(*better_results)
        loss = torch.mean(losses)
        loss.backward() # backpropagate prediction loss, deposit gradients of loss wrt each parameter
        self.optimizer.step() # adjust parameters by gradients collected in backward()
        return losses
    
