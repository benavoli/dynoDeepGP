import torch

class StandardScaler:

    def __init__(self, mean=0.0, std=1.0, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values, dim):
        #dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dim).unsqueeze(dim)
        self.std = torch.std(values, dim=dim).unsqueeze(dim)

    def transform(self, values):
        return  (values - self.mean)/ (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    def inverse_transform(self, values):
        return values* (self.std + self.epsilon) + self.mean