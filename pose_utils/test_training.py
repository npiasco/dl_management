import setlog
import PIL.Image
import torch
import torchvision
import torchvision.transforms.functional as func
import torch.nn.functional as functional
import torch.nn as nn
import torch.autograd as auto
import datasets.custom_quaternion as custom_q
import pose_utils.dlt as utils
import torch.optim as optim
import tqdm


logger = setlog.get_logger(__name__)


def init_net():
    net = nn.Sequential(
        nn.Conv2d(3, 50, 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(50, 100, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(100, 200, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(200, 100, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(100, 50, 4, stride=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(50, 3, 4)
    )

    return net

def module_hook(module, grad_input, grad_out):
    print('module hook')
    print('grad_out', grad_out)

def variable_hook(grad):
    print('variable hook')
    print('grad', grad)


if __name__ == '__main__':
    id = 'frame-000100'

    scale = 0.125

    K = torch.zeros(3, 3)
    K[0, 0] = 585
    K[0, 2] = 320
    K[1, 1] = 585
    K[1, 2] = 240

    K *= 0.5

    K[2, 2] = 1

    root = '/media/nathan/Data/7_Scenes/heads/seq-02/'
    root = '/Users/n.piasco/Documents/Dev/seven_scenes/heads/seq-01/'

    rgb_im = root + id + '.color.png'
    depth_im = root + id + '.depth.png'
    pose_im = root + id + '.pose.txt'

    im = func.to_tensor(func.resize(PIL.Image.open(rgb_im), int(480*scale))).float()
    depth = func.to_tensor(func.resize(PIL.Image.open(depth_im), int(480*scale), interpolation=0),).float()

    depth[depth==65535] = 0
    depth *= 1e-3

    pose = torch.Tensor(4, 4)
    with open(pose_im, 'r') as pose_file_pt:
        for i, line in enumerate(pose_file_pt):
            for j, c in enumerate(line.split('\t')):
                try:
                    pose[i, j] = float(c)
                except ValueError:
                    pass

    rot = pose[0:3, 0:3].numpy()
    quat = custom_q.Quaternion(matrix=rot)
    quat._normalise()
    rot = torch.FloatTensor(quat.rotation_matrix)
    pose[:3, :3] = rot
    '''
    pose =  auto.Variable(pose[:3,:]).cuda()
    im_fwd = auto.Variable(im, requires_grad=True).unsqueeze(0).cuda()
    net = init_net().cuda()
    '''
    pose =  auto.Variable(pose[:3,:])
    im_fwd = auto.Variable(im, requires_grad=True).unsqueeze(0)
    net = init_net()
    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    #net.register_backward_hook(module_hook)
    it = 10000
    n_hyps = 10
    n_pt = 10
    t_loss = list()
    hyps = [[10, 32], [75, 55], [1, 42], [32, 0], [28, 47], [42, 50]]
    for i in tqdm.tqdm(range(it)):
        optimizer.zero_grad()
        output = net(im_fwd)
        #output.register_hook(variable_hook)
        #print(output[0,:,10, 32])
        loss = 0
        for hyp in range(n_hyps):
            pose_net = utils.dlt(utils.draw_hyps(n_pt, width=640*scale, height=480*scale), sceneCoord=output.squeeze(), K=K, grad=True, cuda=False)
           #    pose_net = utils.dlt(hyps, sceneCoord=output.squeeze(), K=K, grad=True, cuda=False)
            #pose_net.register_hook(variable_hook)
            loss += torch.mean(functional.pairwise_distance(pose_net.view(-1,1), pose.view(-1,1)))

        t_loss.append(loss.item())
        if not i%10:
            print('Loss is {}'.format(loss.item()))

        loss.backward()
        optimizer.step()