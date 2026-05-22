import numpy as np
from .base import Module
from scipy.special import erf

class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = np.maximum(input,0)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        mask = (input > 0)
        grad_input = grad_output * mask
        return grad_input


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = 1 / (1 + np.exp(-input))
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        grad_input = grad_output *  (1 / (1 + np.exp(-input))) * (1 - (1 / (1 + np.exp(-input))))
        return grad_input


class GELU(Module):
    """
    Applies element-wise GELU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = input * 0.5 * (1 + erf(input / np.sqrt(2)))
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        cdf = 0.5 * (1 + erf(input / np.sqrt(2)))
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * input**2)
        grad_input = grad_output * (cdf + input * pdf)
        return grad_input


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        max_vals = np.max(input, axis=1, keepdims=True)
        shifted = input - max_vals
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        output = exp_vals / sum_exp
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        y = self.output
        sum_term = np.sum(grad_output * y, axis=1, keepdims=True)
        grad_input = y * (grad_output - sum_term)
        return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # LogSoftmax(x) = log(Softmax(x)) = x - log(sum(exp(x)))
        max_vals = np.max(input, axis=1, keepdims=True)
        shifted = input - max_vals
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        output = shifted - log_sum_exp
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax_output = np.exp(self.output)
        sum_term = np.sum(grad_output, axis=1, keepdims=True)
        grad_input = grad_output - softmax_output * sum_term
        return grad_input
