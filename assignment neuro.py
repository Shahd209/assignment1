import random
def tanh(x):
    return (2 / (1 + 2.71828**(-2*x))) - 1

def initialize_weights():
    return random.uniform(-0.5, 0.5)

i1, i2 = 0.05, 0.10

b1, b2 = 0.5, 0.7

w1, w2, w3, w4 = initialize_weights(), initialize_weights(), initialize_weights(), initialize_weights()
w5, w6, w7, w8 = initialize_weights(), initialize_weights(), initialize_weights(), initialize_weights()

net_h1 = w1 * i1 + w2 * i2 + b1
out_h1 = tanh(net_h1)

net_h2 = w3 * i1 + w4 * i2 + b1
out_h2 = tanh(net_h2)

net_o1 = w5 * out_h1 + w6 * out_h2 + b2
out_o1 = tanh(net_o1)

net_o2 = w7 * out_h1 + w8 * out_h2 + b2
out_o2 = tanh(net_o2)

print("Output of the network:")
print(f"o1: {out_o1}")
print(f"o2: {out_o2}")
