import matplotlib.pyplot as plt
import pose_utils.utils as pc_utils
import torch

pas = 1

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
pc_nn_t = T[i].matmul(pc_nn)
'''
pc_utils.plt_pc(pc.detach().cpu(), ax, pas, 'b', size=100)
pc_utils.plt_pc(pc_nn_t.cpu(), ax, pas, 'c', size=100)

for n, pt in enumerate(pc.detach().t()):
    plt.plot([pt[0],  pc_nn_t[0, n]], [pt[1],  pc_nn_t[1, n]], [pt[2],  pc_nn_t[2, n]], color='g')

centroid = torch.mean(pc[:3, :], -1)

ax.set_xlim([centroid[0].cpu().item() - 1, centroid[0].cpu().item() + 1])
ax.set_ylim([centroid[1].cpu().item() - 1, centroid[1].cpu().item() + 1])
ax.set_zlim([centroid[2].cpu().item() - 1, centroid[2].cpu().item() + 1])
'''
# variables['batch'][0]['K']
K = pc.new_tensor([[[73.1250,  0.0000, 37.0000],
         [ 0.0000, 73.1250, 27.0000],
         [ 0.0000,  0.0000,  1.0000]],

        [[73.1250,  0.0000, 39.0000],
         [ 0.0000, 73.1250, 29.5000],
         [ 0.0000,  0.0000,  1.0000]],

        [[73.1250,  0.0000, 37.5000],
         [ 0.0000, 73.1250, 27.0000],
         [ 0.0000,  0.0000,  1.0000]],

        [[73.1250,  0.0000, 38.0000],
         [ 0.0000, 73.1250, 27.5000],
         [ 0.0000,  0.0000,  1.0000]]])
K[:, :2, :] *= 0.25 # Scale factor
Q = pc.new_tensor([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
reprojected_nn = K[i].matmul(Q.matmul(pc_nn_t))
reprojected_nn[:2, :] /= reprojected_nn[2, :]

reprojected = K[i].matmul(Q.matmul(pc))
reprojected[:2, :] /= reprojected[2, :]

centroid = torch.mean(reprojected[:3, :], -1)

reprojected_nn[:2, :] = torch.round(reprojected_nn[:2, :])
#reprojected  = torch.round(reprojected)

pc_utils.plt_pc(reprojected_nn.cpu(), ax, pas, 'r', size=100, marker='o')
pc_utils.plt_pc(reprojected.detach().cpu(), ax, pas, 'm', size=100, marker='*')

for n, pt in enumerate(reprojected.detach().t()):
    plt.plot([pt[0],  reprojected_nn[0, n]], [pt[1],  reprojected_nn[1, n]], [pt[2],  reprojected_nn[2, n]], color='g')


ax.set_xlim([centroid[0].cpu().item() - 5, centroid[0].cpu().item() + 5])
ax.set_ylim([centroid[1].cpu().item() - 5, centroid[1].cpu().item() + 5])
ax.set_zlim([centroid[2].cpu().item() - 5, centroid[2].cpu().item() + 5])


plt.show()