import numpy as np
import matplotlib.pyplot as plt

train_context = open('../data/train.context').readlines()
train_question = open('../data/train.question').readlines()
train_answer = open('../data/train.answer').readlines()

plt.figure(figsize=(4.5, 4.5))

plt.hist([len(l.split()) for l in train_answer], bins=40)
plt.xlabel('Length')
plt.ylabel('Count')
plt.title('Answer Length in Training Data Set')
plt.show()
