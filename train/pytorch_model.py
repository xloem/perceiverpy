import torch

class PytorchModel:
    # license: gpl-3
    def __init__(self, model, loss_fn = torch.nn.CrossEntropyLoss(reduction='none'), optimizer = None, dev = ('cpu','cuda:0')[torch.cuda.is_available()]):
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
    
