import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        total_sum = np.sum((input - target)**2) / (input.shape[0] * input.shape[1])

        return total_sum

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        grad_input = 2 * (input - target) /  (input.shape[0] * input.shape[1]) 

        return grad_input


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size = input.shape[0]
        self.log_probs = self.log_softmax.compute_output(input)
        
        if self.label_smoothing == 0:
            batch_indices = np.arange(batch_size)
            loss = -np.mean(self.log_probs[batch_indices, target.astype(int)])
        else:
            num_classes = input.shape[1]
            target_one_hot = np.zeros((batch_size, num_classes))
            target_one_hot[np.arange(batch_size), target.astype(int)] = 1
            smooth_targets = (1 - self.label_smoothing) * target_one_hot + self.label_smoothing / num_classes
            loss = -np.sum(smooth_targets * self.log_probs) / batch_size
        
        return loss

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size = input.shape[0]
        softmax = np.exp(self.log_probs)
        
        if self.label_smoothing == 0:
            grad = softmax.copy()
            batch_indices = np.arange(batch_size)
            grad[batch_indices, target.astype(int)] -= 1
        else:
            num_classes = input.shape[1]
            target_one_hot = np.zeros((batch_size, num_classes))
            target_one_hot[np.arange(batch_size), target.astype(int)] = 1
            smooth_targets = (1 - self.label_smoothing) * target_one_hot + \
                           self.label_smoothing / num_classes
            grad = softmax - smooth_targets
        grad = grad / batch_size
        
        return grad
