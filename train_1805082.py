# -*- coding: utf-8 -*-


import numpy as np
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

class Layer:
    """
    The base layer class for all layers

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer

    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name):
        """
        The constructor for the layer class

        Parameters:
            name: The name of the layer
        """

        self.name = name
        self.input_shape = None
        self.output_shape = None

    def forward(self, input):
        """
        The forward pass of the layer

        Parameters:
            x: The input to the layer

        Returns:
            The output of the layer
        """

        raise NotImplementedError

    def backward(self, output_grad):
        # we can use an optimizer instead of learning rate
        """
        The backward pass of the layer

        Parameters:
            grad: The gradient of the loss w.r.t. the output of the layer

        Returns:
            The gradient of the loss w.r.t. the input of the layer
        """

        raise NotImplementedError

def xavier_initialization(m, n):
    """
    Xavier initialization for the weights of a layer

    Parameters:
        m: The number of rows in the weight matrix
        n: The number of columns in the weight matrix

    Returns:
        The initialized weights
    """

    np.random.seed(82)

    std_dev = np.sqrt(2.0 / (m + n))
    return np.random.normal(0, std_dev, size=(m, n))

class DenseLayer(Layer):
    """
    The dense layer class

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer
        weights: The weights of the layer
        bias: The bias of the layer

    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name, input_shape, output_shape):
        """
        The constructor for the dense layer class

        Parameters:
            name: The name of the layer
            input_shape: The shape of the input to the layer
            output_shape: The shape of the output of the layer
        """

        super().__init__(name)
        self.input_shape = input_shape
        self.output_shape = output_shape

        # for adam optimizer
        self.m_wt = None
        self.v_wt = None

        self.m_b = None
        self.v_b = None

        self.t = 0

        self.weights = xavier_initialization(self.output_shape, self.input_shape)
        self.bias = np.zeros((self.output_shape, 1))


    def forward(self, input):
        """
        The forward pass of the layer

        Parameters:
            input: The input to the layer

        Returns:
            The output of the layer
        """

        self.input = input

        # print ("input shape: ", self.input.shape)
        # print ("weights shape: ", self.weights.shape)
        # print ("bias shape: ", self.bias.shape)

        return np.dot(self.weights, self.input) + self.bias


    def backward(self, output_grad):
        """
        The backward pass of the layer

        Parameters:
            output_grad: The gradient of the loss w.r.t. the output of the layer

        Returns:
            The gradient of the loss w.r.t. the input of the layer
        """
        # output_grad: dL/dy (L: loss, y: output of the layer)
        # output_grad: The gradient of the loss w.r.t. the output of the layer
        num_cols = self.input.shape[1]
        self.weights_grad = np.dot(output_grad, self.input.T)/num_cols
        # dL/dB = dL/dy * dy/dB = dL/dy * 1
        self.bias_grad = np.sum(output_grad, axis=1, keepdims=True)/num_cols

        # input_grad: dL/dx
        input_grad = np.dot(self.weights.T, output_grad) / num_cols

        return input_grad

class ActivationLayer(Layer):
    """
    The activation layer class

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer

    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name, activation_func, activation_func_prime):
        """
        The constructor for the activation layer class

        Parameters:
            name: The name of the layer
            activation_func: The activation function of the layer
            activation_func_prime: The derivative of the activation function of the layer
        """

        super().__init__(name)


        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime



    def forward(self, input):
        """
        The forward pass of the layer

        Parameters:
            input: The input to the layer

        Returns:
            The output of the layer
        """

        self.input = input

        return self.activation_func(input)

    def backward(self, output_grad):
        """
        The backward pass of the layer

        Parameters:
            output_grad: The gradient of the loss w.r.t. the output of the layer
            learning_rate: dummy for activation layer

        Returns:
            The gradient of the loss w.r.t. the input of the layer
        """

        return np.multiply(output_grad, self.activation_func_prime(self.input))

class SoftmaxLayer(Layer):
    """
    The softmax layer class

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer

    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name):
        """
        The constructor for the softmax layer class

        Parameters:
            name: The name of the layer
        """

        super().__init__(name)

    def forward(self, input):
        """
        The forward pass of the layer

        Parameters:
            input: The input to the layer

        Returns:
            The output of the layer
        """

        # num_cols = input.shape[1]

        # for i in range(num_cols):
        #     input[:, i] = input[:, i] - np.max(input[:, i])
        #     # to avoid overflow

        #     numerator = np.exp(input[:, i])
        #     denominator = np.sum(numerator)
        #     input[:, i] = numerator / denominator

        # self.output = input
        # return self.output

        input -= np.max(input, axis=0, keepdims=True)

        # Compute the numerator and denominator
        numerator = np.exp(input)
        denominator = np.sum(numerator, axis=0, keepdims=True)

        # Compute the softmax probabilities
        self.output = numerator / denominator

        return self.output


        # z = input - max(input, axis = 1)
        # numerator = np.exp(z)
        # denominator = np.sum(numerator)
        # self.output =  numerator / denominator
        # return self.output


    def backward(self, output_grad, y_label):
        """
        The backward pass of the layer

        Parameters:
            output_grad: The gradient of the loss w.r.t. the output of the layer
            learning_rate: dummy for softmax layer

        Returns:
            The gradient of the loss w.r.t. the input of the layer
        """


        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_grad)
        # * is element-wise multiplication, not matrix multiplication

        return self.output - y_label

class DropoutLayer(Layer):
    """
    The dropout layer class

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer
        dropout_rate: The dropout rate of the layer

    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name, dropout_rate):
        """
        The constructor for the dropout layer class

        Parameters:
            name: The name of the layer
            dropout_rate: The dropout rate of the layer
        """

        super().__init__(name)
        self.dropout_rate = dropout_rate
        self.dropout_mask = None


    def forward(self, input, training = True):
        """
        The forward pass of the layer

        Parameters:
            input: The input to the layer

        Returns:
            The output of the layer
        """

        if not training:
            return input

        self.input = input

        np.random.seed(82)
        D1 = np.random.rand(*input.shape) < (1 - self.dropout_rate)
        # D1 is a binary mask
        # print ("D1: \n", D1)

        self.dropout_mask = D1 / (1 - self.dropout_rate)

        # print ("dropout mask: \n", self.dropout_mask)

        return np.multiply(input, self.dropout_mask) / (1 - self.dropout_rate)
        # h/(1-p)

    def backward(self, output_grad):
        """
        The backward pass of the layer

        Parameters:
            output_grad: The gradient of the loss w.r.t. the output of the layer
            learning_rate: dummy for dropout layer

        Returns:
            The gradient of the loss w.r.t. the input of the layer
        """

        return np.multiply(output_grad, self.dropout_mask) / (1 - self.dropout_rate)

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class ReLUActivationLayer(ActivationLayer):
    """
    The ReLU activation layer class

    Attributes:
        name: The name of the layer
        input_shape: The shape of the input to the layer
        output_shape: The shape of the output of the layer

    Methods:
        forward: The forward pass of the layer
        backward: The backward pass of the layer
    """

    def __init__(self, name):
        """
        The constructor for the ReLU activation layer class

        Parameters:
            name: The name of the layer
        """



        # super().__init__(name, lambda x: np.maximum(x, 0), lambda x: np.where(x > 0, 1, 0))
        super().__init__(name, relu, relu_derivative)

"""Cross Entropy loss"""

def cross_entropy_loss(y, y_pred):
    """
    The cross entropy loss function

    Parameters:
        y: The ground truth
        y_pred: The predictions

    Returns:
        The loss
    """


    epsilon = 1e-15  # small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # limits the values in the y_pred array to be within the range
    # [epsilon, 1 - epsilon]. This ensures that the predicted probabilities are not exactly zero or one

    # print ("in cross entropy loss")
    # print ("y size: ", y.size)
    # print ("y_pred: ", y_pred)

    return -np.sum(np.multiply(y, np.log(y_pred))) / y.size



def cross_entropy_loss_prime(y, y_pred):
    """
    The derivative of the cross entropy loss function with respect to the y_pred

    Parameters:
        y: The ground truth
        y_pred: The predictions

    Returns:
        The derivative of the loss w.r.t. the predictions
    """


    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # print ("in cross entropy loss prime")
    # print ("y size: ", y.size)
    # print ("y_pred: ", y_pred)

    return -np.divide(y, y_pred) / y.size

"""Optimizer"""

class AdamOptimizer:
    """
    The Adam optimizer class

    Attributes:
        learning_rate: The learning rate
        beta1: The beta1 parameter
        beta2: The beta2 parameter
        epsilon: The epsilon parameter

    Methods:
        update: Updates the parameters
    """

    def __init__(self, learning_rate=5e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


    def update(self, layer):
        """
        Updates the parameters

        Parameters:
            layer: The layer to update

        """

        dw = layer.weights_grad
        db = layer.bias_grad

        if layer.m_wt is None:

            layer.m_wt = np.zeros_like(dw)
            layer.v_wt = np.zeros_like(dw)

            layer.m_b = np.zeros_like(db)
            layer.v_b = np.zeros_like(db)

        layer.t += 1

        layer.m_wt = self.beta1 * layer.m_wt + (1 - self.beta1) * dw
        layer.v_wt = self.beta2 * layer.v_wt + (1 - self.beta2) * (dw ** 2)

        m_hat_wt = layer.m_wt / (1 - self.beta1 ** layer.t)
        v_hat_wt = layer.v_wt / (1 - self.beta2 ** layer.t)

        layer.weights -= self.learning_rate * m_hat_wt / (np.sqrt(v_hat_wt) + self.epsilon)


        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * db
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (db ** 2)

        m_hat_b = layer.m_b / (1 - self.beta1 ** layer.t)
        v_hat_b = layer.v_b / (1 - self.beta2 ** layer.t)

        layer.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

"""Moodle instructed dataset loading"""

# main function

def main():


    

    import torchvision.datasets as ds
    from torchvision import transforms


    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)


    independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                                train=False,
                                transform=transforms.ToTensor())

    # Split the train-validation dataset as 85%-15% to form your train set and validation set using sklearn.model_selection.train_test_split

    from sklearn.model_selection import train_test_split

    train_dataset, validation_dataset = train_test_split(train_validation_dataset, test_size=0.15, random_state=82)

    """Visualization"""

    def show_images(dataset, num_images=6):
        """
        Show the first num_images images of the dataset

        Parameters:
            dataset: The dataset to show the images from
            num_images: The number of images to show
        """

        # Create a figure to display the images
        fig = plt.figure()

        # Loop over the first num_images images in the dataset
        for i in range(num_images):
            # Get the image and its label
            image, label = dataset[i]

            # The image needs to be transposed to be displayed correctly
            image = image.transpose(0,2).transpose(0,1)

            # Display the image
            plt.subplot(1, num_images, i + 1)
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'Label: {label}')
            plt.axis('off')

        # Display the figure
        plt.show()

    show_images(train_dataset)

    """Preprocessing"""

    def preprocess_data(x, y):

        # normalize x
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x.reshape(x.shape[0], 28 * 28))
        x = x.reshape(x.shape[0], 28 * 28, 1)

        # reduce each value in y by 1, so that the range of y is [0, 25]
        y = y - 1

        # convert class labels to one-hot encoded, should have shape (?, 26, 1)
        # 26 classes: 26 letters, for example, C is 3rd letter, then C is represented as [0, 0, 1, 0, 0, ...]
        y = np.eye(26)[y]

        y = y.reshape(y.shape[0], 26, 1)

        return x, y

    X = np.array([np.array(x[0]).flatten() for x in train_dataset])
    Y = np.array([x[1] for x in train_dataset])

    X_validation = np.array([np.array(x[0]).flatten() for x in validation_dataset])
    Y_validation = np.array([x[1] for x in validation_dataset])

    X_test = np.array([np.array(x[0]).flatten() for x in independent_test_dataset])
    Y_test = np.array([x[1] for x in independent_test_dataset])


    X, Y = preprocess_data(X, Y)
    X_validation, Y_validation = preprocess_data(X_validation, Y_validation)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    """Train the model"""

    network = [
        DenseLayer("Dense1", X.shape[1], 1024),
        ReLUActivationLayer("ReLU1"),
        DropoutLayer("Dropout1", 0.3),
        DenseLayer("Dense2", 1024, 26),
        SoftmaxLayer("Softmax"),
    ]


    learning_rate = 5e-4
    epochs = 4

    adam_optimizer = AdamOptimizer(learning_rate=learning_rate)

    for epoch in range(epochs):
        loss = 0

        # use minibatch gradient descent
        batch_size = 1024
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        # print ("num batches: ", num_batches)

        # zip the data and shuffle
        zipped_data = list(zip(X, Y))

        random.seed(82)
        random.shuffle(zipped_data)

        for i in tqdm(range(num_batches)):
            # print ("batch: ", i)

            # get the minibatch
            batching_time = time.time()
            batch = zipped_data[i * batch_size : (i + 1) * batch_size]

            # unzip the minibatch
            X_batch, Y_batch = zip(*batch)

            X_batch = np.reshape(X_batch, (len(X_batch), 28 * 28))
            X_batch = np.array(X_batch).T
            Y_batch = np.reshape(Y_batch, (len(Y_batch), 26))
            Y_batch = np.array(Y_batch).T

            batching_time = time.time() - batching_time
            # print ("batching time: ", batching_time)

            output = X_batch
            forward_time_start = time.time()
            for layer in network:
                # print ("forward layer: ", layer.name)
                output = layer.forward(output)
                # print ("output: \n", output)

            forward_time = time.time() - forward_time_start

            # print("forward time: ", forward_time)

            loss_grad_time_start = time.time()

            loss += cross_entropy_loss(Y_batch, output)
            output_grad = cross_entropy_loss_prime(Y_batch, output)

            # print ("loss: \n", loss)

            loss_grad_time = time.time() - loss_grad_time_start

            # print("loss grad time: ", loss_grad_time)

            backward_time_start = time.time()
            for layer in reversed(network):
                # print ("backward layer: ", layer.name)

                if isinstance(layer, SoftmaxLayer):
                    output_grad = layer.backward(output_grad, Y_batch)

                else:
                    output_grad = layer.backward(output_grad)

                # print ("output grad: \n", output_grad)

                if isinstance(layer, DenseLayer):
                    # normal gradient descent
                    # layer.weights -= learning_rate * layer.weights_grad
                    # layer.bias -= learning_rate * layer.bias_grad

                    # adam optimizer
                    adam_optimizer.update(layer=layer)

            backward_time = time.time() - backward_time_start
            # print("backward time: ", backward_time)


        print(f"Epoch {epoch + 1}: Loss = {loss/num_batches:.5f}")

    def test_model(network, X_test, Y_test):

        # test the model

        # print ("X_test: ", X_test)
        # print ("X_test shape:", X_test.shape)

        # print ("Y_test: ", Y_test)
        # print ("Y_test shape:", Y_test.shape)


        output_list = []
        for x in X_test:
            output = x
            for layer in network:
                if isinstance(layer, DropoutLayer):
                    output = layer.forward(output, training=False)

                else:
                    output = layer.forward(output)

            output_list.append(output)


        # print ("output: ", output_list)

        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

        y_true = np.argmax(Y_test, axis=1)
        y_pred = np.argmax(output_list, axis=1)

        # print ("y_true: ", y_true)
        # print ("y_pred: ", y_pred)

        print ("accuracy: ", accuracy_score(y_true, y_pred))
        print ("precision: ", precision_score(y_true, y_pred, average='macro'))
        print ("recall: ", recall_score(y_true, y_pred, average='macro'))
        print ("f1 score: ", f1_score(y_true, y_pred, average='macro'))

        # todo fix
        # print ("confusion matrix: ")
        # print (confusion_matrix(y_true, y_pred))

        conf_mat = confusion_matrix(y_true, y_pred)

        # Create a heatmap for the confusion matrix
        import seaborn as sns
        plt.figure(figsize=(14, 14))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)


        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    """Training accuracy"""

    test_model(network, X, Y)

    """Validation dataset for model selection"""

    test_model(network, X_validation, Y_validation)

    """Independent test dataset"""

    test_model(network, X_test, Y_test)

    """Save in pickle format"""

    import pickle

    with open('model_1805082.pickle', 'wb') as model_file:
        pickle.dump(network, model_file)


if __name__ == '__main__':
    main()

