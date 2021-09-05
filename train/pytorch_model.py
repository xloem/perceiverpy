import torch

import os
import weakref

def name2path(*names):
    return os.path.join('.', 'models', *names[:-1], names[-1] + '.pystate')

def path2name(path):
    return os.path.splitext(os.path.basename(path))[0]

default_device = torch.device(('cpu','cuda:0')[torch.cuda.is_available()])

def stack2name(maxdepth=2):
    import inspect
    result = []
    depth = 0
    lastfile = None
    for frame in inspect.stack()[1:]:
        name = path2name(frame.filename)
        if '<' not in name:
            if name != lastfile:
                lastfile = name
                depth += 1
                if depth > maxdepth:
                    break
            #name += '.' + str(frame.lineno)
            if '<' not in frame.function:
                name += '.' + frame.function
            result.append(name)
    result.reverse()
    return '-'.join(result)

def dict2name(dict):
    def shorten(val):
        if type(val) is str:
            result = ''
            lastchr = ''
            for chr in val:
                if lastchr == '' or lastchr == '_':
                    result += chr
                lastchr = chr
            return result
        elif type(val) is bool:
            return 'fT'[val]
        else:
            return str(val)
    return ''.join((
        shorten(key) + shorten(value)
        for key, value in dict.items()
    ))

_models = weakref.WeakValueDictionary()

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
        if hasattr(model, 'name') and hasattr(model, 'dev'):
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

    def update(self, *better_results):
        better_results = torch.stack([self._tensor(result, None) for result in better_results])
        better_results = better_results.to(self.dev)

        losses = self.loss_fn(self.last_predictions, better_results)
        loss = torch.mean(losses)

        loss.backward() # backpropagate prediction loss, deposit gradients of loss wrt each parameter
        self.optimizer.step() # adjust parameters by gradients collected in backward()
        return losses
    
