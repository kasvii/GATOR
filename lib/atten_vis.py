from typing import Set
import seaborn
import matplotlib.pyplot as plt
import numpy as np
# seaborn.set_context(context="talk")
seaborn.set()

def draw(data, ax):
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=30)
    axx = seaborn.heatmap(data, 
                    square=True, vmin=-2, vmax=2,
                    cbar=False, annot=False, ax=ax, cmap="Blues")
    # cbar = axx.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=15)
    # seaborn.heatmap(data, annot=True) vmin=-0.0001, vmax=0.2,  vmin=-5, vmax=2,    vmin=-4, vmax=4, , cbar_kws={"ticks":np.arange(-4,5,1), "format":"%.0f"}
    
# ori_atten.npy
# softmax_atten.npy

# spatial_attn.npy
# edge_attn.npy
# attn_bias.npy
# before_softmax_atten.npy
# edg_adj.npy
attn_bias = np.load('./attn_bias.npy')
'''
fig, axs = plt.subplots(figsize=(6, 5))
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 15)
draw(attn_bias.data, ax=axs)

plt.savefig('./edg_adj.png')
'''

fig, axs = plt.subplots(1,8, figsize=(40, 5))
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 15)
for h in range(8):
    draw(attn_bias[h].data, ax=axs[h])
    axs[h].tick_params(labelsize='large')
    axs[h].set_title('Head {}'.format(h), fontsize = 18, pad =10 )
    # axs[h].set_xticks(fontsize=12 )
# for ax in axs:
    
plt.savefig('./attn_bias-0_7-2_2-large.png')

# plt.show()
    