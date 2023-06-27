import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from matplotlib import pyplot as plt

from utils import mkdir_if_missing

mkdir_if_missing(os.path.join(BASE_DIR, "plots"))

# x = [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
x = [0, 1, 2, 3, 4, 5, 6]
lambda1_score = [0.395, 0.405, 0.403, 0.406, 0.411, 0.401, 0.395]
lambda3_score = [0.399, 0.400, 0.402, 0.407, 0.411, 0.409, 0.403]
plt.figure(dpi=1200)

plt.xlabel(xlabel="$\lambda_{1}$", fontsize=20)
# plt.xlabel(xlabel='$\lambda_{3}$', fontsize=20)
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["0.01", "0.05", "0.1", "0.5", "1", "2", "5"], fontsize=18)
plt.yticks(ticks=[0.390, 0.395, 0.400, 0.405, 0.410, 0.415],
           labels=["0.390", "0.395", "0.400", "0.405", "0.410", "0.415"], fontsize=18)
plt.ylabel("mAP@all", fontsize=17)

plt.plot(x, lambda1_score, '-',linewidth=3)
# plt.plot(x, lambda3_score, "-", linewidth=3)
# plt.legend()
plt.ylim(0.39, 0.415)
plt.tight_layout()
plt.grid(linestyle='--')
plt.savefig(os.path.join(BASE_DIR, "plots", "lambda1.png"))
# plt.savefig(os.path.join(BASE_DIR, "plots", "lambda3.png"))
plt.show()
