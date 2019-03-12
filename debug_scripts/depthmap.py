import matplotlib.pyplot as plt
import torchvision as torchvis
import torch
import math

ccmap = plt.get_cmap('jet', lut=1024)
d_fact= 1

modality = pc.new_zeros(1, 1, 56//d_fact, 56//d_fact)
modality_ref = pc.new_zeros(1, 1, 56//d_fact, 56//d_fact)
modality_idx = pc.new_zeros(1, 1, 56//d_fact, 56//d_fact)
modality_ref_idx = pc.new_zeros(1, 1, 56//d_fact, 56//d_fact)

modality[:, :, r_n_rep_pc[1, :].long(), r_n_rep_pc[0, :].long()] = rep_pc[2, :]
#modality_ref[:, :, r_n_rep_pc_nn_t[1, :].long(), r_n_rep_pc_nn_t[0, :].long()] = rep_pc_nn_t[2, :]
modality_idx[:, :, r_n_rep_pc[1, idx].long(), r_n_rep_pc[0, idx].long()] = rep_pc[2, idx]
modality_ref_idx[:, :, r_n_rep_pc_nn_t[1, idx].long(), r_n_rep_pc_nn_t[0, idx].long()] = rep_pc_nn_t[2, idx]

d_mod = torch.abs(modality_idx - modality_ref_idx)*1
d_mod_idx = torch.abs(filter(filter(modality_idx - modality_ref_idx)))*10
for nfilter in range(10):
    d_mod_idx[:, :, r_n_rep_pc[1, idx].long(), r_n_rep_pc[0, idx].long()] = torch.abs(modality_idx[:, :, r_n_rep_pc[1, idx].long(), r_n_rep_pc[0, idx].long()] - modality_ref_idx[:, :, r_n_rep_pc[1, idx].long(), r_n_rep_pc[0, idx].long()])*10
    d_mod_idx = filter(d_mod_idx)
modality_corr = modality - filter(modality_idx - modality_ref_idx)

fig = plt.figure(1)
images_batch = torch.cat((modality.detach().cpu(), modality_corr.detach().cpu(),
                          modality_idx.detach().cpu(), modality_ref_idx.detach().cpu(),
                          d_mod_idx.detach().cpu(), d_mod.detach().cpu()))
grid = torchvis.utils.make_grid(images_batch, nrow=2)
plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
plt.colorbar()

plt.show()


'''
import matplotlib.pyplot as plt
import torchvision as torchvis
import torch
import math

ccmap = plt.get_cmap('jet', lut=1024)

mod = new_depth_maps.unsqueeze(0)
trumode = torch.nn.functional.interpolate(sample['depth'].unsqueeze(0), scale_factor=0.5)

im = new_image.unsqueeze(0)
truim = torch.nn.functional.interpolate(sample['rgb'].unsqueeze(0), scale_factor=0.5)



fig = plt.figure(1)
images_batch = torch.cat((mod, trumode,))
grid = torchvis.utils.make_grid(images_batch, nrow=2)
plt.imshow(grid.numpy().transpose(1, 2, 0)[:, :, 0], cmap=ccmap)
plt.colorbar()



fig = plt.figure(2)
images_batch = torch.cat((im, truim,))
grid = torchvis.utils.make_grid(images_batch, nrow=2)
plt.imshow(grid.numpy().transpose(1, 2, 0))
plt.show()
'''