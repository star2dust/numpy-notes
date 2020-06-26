# Numpy and Matplotlib

## Install Requirements

```shell
pip install numpy
pip install matplotlib
```

## Tutorials

See [star2dust.github.io/numpy-notes/](https://star2dust.github.io/numpy-notes/). (Chinese)

## Examples

- Calculate `relu` and `sigmoid` function.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# fig size
plt.figure(figsize=(8, 3))

# from -10.0 to 10.0 with interval = 0.1
x = np.arange(-10, 10, 0.1)
# calculate Sigmoid function
s = 1.0 / (1 + np.exp(- x))

# calculate ReLU function
y = np.clip(x, a_min = 0., a_max = None)

#########################################################
# plot

# two subfigures, Sigmoid in the left
f = plt.subplot(121)
# plot curve
plt.plot(x, s, color='r')
# add text
plt.text(-5., 0.9, r'$y=\sigma(x)$', fontsize=13)
# set x and y axis
currentAxis=plt.gca()
currentAxis.xaxis.set_label_text('x', fontsize=15)
currentAxis.yaxis.set_label_text('y', fontsize=15)

# ReLU in the right
f = plt.subplot(122)
# plot curve
plt.plot(x, y, color='g')
# add text
plt.text(-3.0, 9, r'$y=ReLU(x)$', fontsize=13)
# set x and y axis
currentAxis=plt.gca()
currentAxis.xaxis.set_label_text('x', fontsize=15)
currentAxis.yaxis.set_label_text('y', fontsize=15)

plt.show()
```

<img src="examples/relu_sigmoid.png" alt="relu_sigmoid" width="350"/>