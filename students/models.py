import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        while True:
            mistakes = 0
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                ground_truth = nn.as_scalar(y)
                
                if prediction != ground_truth:
                    self.w.update(x, ground_truth)
                    mistakes += 1
            if mistakes == 0:
                break
                    

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.hidden_size = 200
        self.lr = 0.01
        self.batch_size = 10
        self.W1 = nn.Parameter(1, self.hidden_size)
        self.B1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 1)
        self.B2 = nn.Parameter(1, 1)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        h1 = nn.Linear(x, self.W1)
        h1_bias = nn.AddBias(h1, self.B1)
        h1_relu = nn.ReLU(h1_bias)
        
        output = nn.Linear(h1_relu, self.W2)
        output = nn.AddBias(output, self.B2)
        
        return output
        

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                gradients = nn.gradients(loss, [self.W1, self.B1, self.W2, self.B2])
                grad_W1, grad_B1, grad_W2, grad_B2 = gradients
                
                self.W1.update(grad_W1, -self.lr)
                self.B1.update(grad_B1, -self.lr)
                self.W2.update(grad_W2, -self.lr)
                self.B2.update(grad_B2, -self.lr)
                
            total_loss = 0
            count = 0
            for x, y in dataset.iterate_once(self.batch_size):
                total_loss += nn.as_scalar(self.get_loss(x, y))
                count += 1
            avg_loss = total_loss/count
            if avg_loss < 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.hidden_size = 125
        self.lr = 0.07
        self.batch_size = 150
        
        self.W1 = nn.Parameter(784, self.hidden_size)
        self.B1 = nn.Parameter(1, self.hidden_size)
        
        self.W2 = nn.Parameter(self.hidden_size, 10)
        self.B2 = nn.Parameter(1, 10)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        h1 = nn.Linear(x, self.W1)
        h1 = nn.AddBias(h1, self.B1)
        h1 = nn.ReLU(h1)
        
        output = nn.Linear(h1, self.W2)
        output = nn.AddBias(output, self.B2)
        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                gradients = nn.gradients(loss, [self.W1, self.B1, self.W2, self.B2])
                grad_W1, grad_B1, grad_W2, grad_B2 = gradients
                
                self.W1.update(grad_W1, -self.lr)
                self.B1.update(grad_B1, -self.lr)
                self.W2.update(grad_W2, -self.lr)
                self.B2.update(grad_B2, -self.lr)
            accuracy = dataset.get_validation_accuracy()
            if accuracy > 0.97:
                break
