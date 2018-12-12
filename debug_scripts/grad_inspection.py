import torch

networks = ['Main', 'Deconv']
print('Gradiant inspection...')
for nets_name in networks:
    for params in self.networks[nets_name].get_training_layers():
        for param in params['params']:
            print(param.size(), torch.max(param.grad), torch.min(param.grad))

print('Clipping...')
for nets_name in networks:
    for params in self.networks[nets_name].get_training_layers():
        for param in params['params']:
            torch.nn.utils.clip_grad_value_(param, 1)

print('Gradiant inspection (after clipping)...')
for nets_name in networks:
    for params in self.networks[nets_name].get_training_layers():
        for param in params['params']:
            print(torch.max(param.grad))
            print(torch.min(param.grad))
