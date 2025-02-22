import random
def tanh(x):
    return (2 / (1 + exp(-2 * x))) - 1
def exp(x, terms=10):
    """ Compute exponential using a Taylor series approximation """
    result = 1
    factorial = 1
    power = 1
    for i in range(1, terms):
        factorial *= i
        power *= x
        result += power / factorial
    return result
def forward_pass(i1, i2, w, b1, b2):
    
    net_h1 = w[0] * i1 + w[1] * i2 + b1
    net_h2 = w[2] * i1 + w[3] * i2 + b1
    out_h1 = tanh(net_h1)
    out_h2 = tanh(net_h2)
    
    net_o1 = w[4] * out_h1 + w[5] * out_h2 + b2
    net_o2 = w[6] * out_h1 + w[7] * out_h2 + b2
    out_o1 = tanh(net_o1)
    out_o2 = tanh(net_o2)
    
    return out_o1, out_o2

i1, i2 = 0.05, 0.1
random.seed(42)
w = [random.uniform(-0.5, 0.5) for _ in range(8)]

b1, b2 = 0.5, 0.7
o1, o2 = forward_pass(i1, i2, w, b1, b2)
print("Output of the network:")
print("O1:", o1)
print("O2:", o2)
