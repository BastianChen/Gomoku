import numpy as np

# a = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1]]).reshape(4, 3)
# print(a)
# b = a.flatten().nonzero()
# print(b)
# print(b[0])
# c = a.take(b[0])
# print(c)
# d = c.prod(axis=-1)
# print(d)

a = list(range(1))
print(a)
a.remove(0)
print(a)
if len(a) == 0:
    print('yes')
if not len(a):
    print('yes')
