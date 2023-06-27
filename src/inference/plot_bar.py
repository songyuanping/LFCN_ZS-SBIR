import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from matplotlib import pyplot as plt

from utils import mkdir_if_missing

mkdir_if_missing(os.path.join(BASE_DIR, "plots"))

# data = [random.random()for i in range(15)]+[10]+[random.random()for i in range(15)]+[5,1,8]*1+[random.random()for i in range(15)]
data = [0.5for i in range(15)]+[0.5]+[0.5for i in range(15)]+[3]*1+[0.5for i in range(15)]

plt.bar(range(len(data)), data,color='royalblue')
plt.xticks([])
plt.yticks([])

# plt.legend()
# plt.ylim(0.39, 0.415)
plt.tight_layout()
plt.axis('off')
plt.savefig(os.path.join(BASE_DIR, "plots", "label_bar.png"))
plt.show()
