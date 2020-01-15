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

# a = list(range(1))
# print(a)
# a.remove(0)
# print(a)
# if len(a) == 0:
#     print('yes')
# if not len(a):
#     print('yes')

# a = [1, 2, 3, 4, 5]
# b = np.random.rand(len(a))
# c = list(zip(a, b))
# print(c)
# d = list([a, b])
# print(d)

# a = {}
# a[0] = 1
# a[1] = 2
# a[2] = 0
# print(a)
# b = max(a.items(), key=lambda value: abs(value[1]))
# print(a.items())
# print(b)

# from utils import softmax
#
# visits = [2000, 1500, 0, 0, 0]
# # visits = np.array(visits, dtype=np.float32)
# # print(np.mean(visits))
# # print(np.max(visits))
# # visits = (visits - np.mean(visits)) / np.max(visits)
# # print(visits)
# act_probs = softmax(1.0 / 1e-3 * np.log(np.array(visits) + 1e-10))
# # act_probs = softmax(visits)
# print(act_probs)

# action_probs = np.zeros(6 * 6, dtype=np.float32)
# print(action_probs)
# a = [0, 2, 3]
# b = [1, 2, 3]
# action_probs[list(a)] = b
# print(action_probs)

b = 1
a = b
a = 2
print(b)
print(a)
