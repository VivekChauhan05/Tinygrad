import math
class Value:
    """ Stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        # Initialize the Value object with data, gradient, and related properties.
        self.data = data
        self.grad = 0
        # Internal variables used for autograd graph construction.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # The operation that produced this node, for debugging, etc.

    def __add__(self, other):
        # Overload the addition operator to create a new Value object for addition.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Define the backward pass for addition.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        # Overload the multiplication operator to create a new Value object for multiplication.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Define the backward pass for multiplication.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # Overload the power operator to create a new Value object for exponentiation.
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Define the backward pass for exponentiation.
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        # Implement the sigmoid activation function.
        exp_val = math.exp(-self.data)
        out = Value(1 / (1 + exp_val), (self,), 'Sigmoid')

        def _backward():
            # Define the backward pass for the sigmoid function.
            s = out.data
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def linear(self):
        # Implement a linear activation (identity function).
        out = Value(self.data, (self,), 'Linear')

        def _backward():
            # The gradient of the linear function is 1.
            self.grad += 1 * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        # Implement the ReLU (Rectified Linear Unit) activation function.
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # Define the backward pass for ReLU.
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        # Calculate the hyperbolic tangent (tanh) of the input value.
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
   
       # Create a new Value object to store the result and track the computation graph.
        out = Value(t, (self, ), 'tanh')

        def _backward():
           # Define the backward pass for the tanh function.
           # The derivative of tanh is (1 - tanh^2).
           self.grad += (1 - t**2) * out.grad
        out._backward = _backward
   
        return out
        
    def backward(self):
        # Perform the backward pass to compute gradients using the chain rule.

        # Topological order all of the children in the graph.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize the gradient to 1 for the current Value object.
        self.grad = 1

        # Go one variable at a time and apply the chain rule to get its gradient.
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # Implement the negation operation (-self).
        return self * -1

    def __radd__(self, other):  # Implement the reverse addition (other + self).
        return self + other

    def __sub__(self, other):  # Implement subtraction (self - other).
        return self + (-other)

    def __rsub__(self, other):  # Implement reverse subtraction (other - self).
        return other + (-self)

    def __rmul__(self, other):  # Implement reverse multiplication (other * self).
        return self * other

    def __truediv__(self, other):  # Implement true division (self / other).
        return self * other ** -1

    def __rtruediv__(self, other):  # Implement reverse true division (other / self).
        return other * self ** -1

    def __repr__(self):
        # Define the string representation of the Value object.
        return f"Value(data={self.data}, grad={self.grad})"
