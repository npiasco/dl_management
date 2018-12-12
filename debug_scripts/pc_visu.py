import matplotlib.pyplot as plt
import pose_utils.utils as pc_utils
import torch
from mpl_toolkits.mplot3d import axes3d


pas = 10

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

pc1_ = pc_ref.detach()
pc2_ = pc_to_align.detach()

centroid = torch.mean(pc1_[:3, :], -1)

pc_utils.plt_pc(pc2_.cpu(), ax, pas, 'r', size=100, marker='+')
pc_utils.plt_pc(pc1_.detach().cpu(), ax, pas, 'm', size=100, marker='*')

for n in range(0, pc1_.size(-1), pas):
    plt.plot([pc1_[0, n],  pc2_[0, n]], [pc1_[1, n],  pc2_[1, n]], [pc1_[2, n],  pc2_[2, n]], color='g')


ax.set_xlim([centroid[0].cpu().item() - 5, centroid[0].cpu().item() + 5])
ax.set_ylim([centroid[1].cpu().item() - 5, centroid[1].cpu().item() + 5])
ax.set_zlim([centroid[2].cpu().item() - 5, centroid[2].cpu().item() + 5])


plt.show()