from collections import namedtuple
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm_notebook
from tqdm import tqdm as tqdm
from model import NNModel
from utils import gram_matrix, recover_image, tensor_normalizer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

IMAGE_SIZE = 256
BATCH_SIZE = 64
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e5
LOG_INTERVAL = 200  # Log Results with interval of LOG_INTERVAL
REGULARIZATION = 1e-7
LR = 1e-3
DATASET = "/scratch/jv1589/Project/perceptual-losses/coco"
SEED=2000

torch.cuda.manual_seed(SEED)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)


def get_style_image():
    """
    Returns the neural style base.
    """
    STYLE_IMAGE = "images/inputs/style/starry_night.jpg"
    style_img = Image.open(STYLE_IMAGE).convert('RGB')
    style_img_tensor = transforms.Compose([
        transforms.ToTensor(),
        tensor_normalizer()]
    )(style_img).unsqueeze(0)

    # Convert to cuda
    style_img_tensor = style_img_tensor.cuda()

    return style_img_tensor


def save_debug_image(tensor_orig, tensor_transformed, filename):
    """
    Converts a tensor into an image and saves it.
    """
    result = Image.fromarray(recover_image(tensor_transformed.cpu().numpy())[0])
    orig = Image.fromarray(recover_image(tensor_orig.cpu().numpy())[0])
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))

    new_im.paste(orig, (0,0))
    new_im.paste(result, (result.size[0] + 5,0))
    new_im.save(filename)


def get_train_data():
    """
    Generates training data
    """
    transform = transforms.Compose([transforms.Scale(IMAGE_SIZE),
                                    transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    tensor_normalizer()])

    train_dataset = datasets.ImageFolder(DATASET, transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

    return train_dataset, train_loader

def get_loss_model():
    """
    Gets the NN Model
    """
    vgg_model = vgg.vgg16(pretrained=True)
    vgg_model.cuda()

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    return loss_network


def plot_graph(style_loss_list, content_loss_list):
    """
    Plots Style_Loss and Content_Loss graphs
    """
    with open('/scratch/jv1589/Project/perceptual-losses/content_loss.txt', 'w') as f:
        for item in content_loss_list:
            f.write("%s\n" % item)

    with open('/scratch/jv1589/Project/perceptual-losses/style_loss.txt', 'w') as f2:
        for item in style_loss_list:
            f2.write("%s\n" % item)


def evaluate_style_transfer(nnmodel, filename):
    """
    Given a content image and the required model, perform the style transfer
    """

    img = Image.open(filename).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor(),
                                    tensor_normalizer()])
    img_tensor = transform(img).unsqueeze(0)

    print(img_tensor.size())

    img_tensor = img_tensor.cuda()

    img_output = nnmodel(Variable(img_tensor, volatile=True))
    plt.imsave('kinkaku.png', (recover_image(img_output.cpu().detach().numpy())[0]))


def train_model():
    """
    Trains the neural network model
    """
    loss_network = get_loss_model()
    print("Generating the dataset")
    train_dataset, train_loader = get_train_data()
    print("{} ".format(len(train_dataset)))


    style_img_tensor = get_style_image()

    style_loss_list = []
    content_loss_list = []

    style_loss_features = loss_network(Variable(style_img_tensor, volatile=True))
    gram_style = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features]

    nnmodel = NNModel()
    mse_loss = torch.nn.MSELoss()

    nnmodel.cuda()  # Convert to CUDA


    optimizer = Adam(nnmodel.parameters(), LR)
    nnmodel.train()


    for epoch in range(2):
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in tqdm_notebook(enumerate(train_loader), total=len(train_loader)):
            n_batch = len(x)
            count += n_batch

            optimizer.zero_grad()

            x = Variable(x)
            x = x.cuda()
            y = nnmodel(x)
            xc = Variable(x.data, volatile=True)

            features_y = loss_network(y)
            features_xc = loss_network(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = CONTENT_WEIGHT * mse_loss(features_y[1], f_xc_c)


            style_loss = 0.

            for m in range(len(features_y)):
                g_style = gram_style[m]
                gram_y = gram_matrix(features_y[m])
                style_loss += STYLE_WEIGHT * mse_loss(gram_y, g_style.expand_as(gram_y))

            total_loss = content_loss + style_loss  # Sum of losses
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss
            agg_style_loss += style_loss

            if (batch_id + 1) % LOG_INTERVAL == 0:
                print("Content-Loss: {:.6f}  Style-Loss: {:.6f}  Total Loss: {:.6f}".format(agg_content_loss / LOG_INTERVAL, agg_style_loss / LOG_INTERVAL, (agg_content_loss + agg_style_loss) / LOG_INTERVAL))

                style_loss_list.append(agg_style_loss / LOG_INTERVAL)
                content_loss_list.append(agg_content_loss / LOG_INTERVAL)

                agg_content_loss = 0
                agg_style_loss = 0
                nnmodel.eval()
                y = nnmodel(x)
                save_debug_image(x.data, y.data, "debug/{}_{}.png".format(epoch, count))  # Save image
                nnmodel.train()

    plot_graph(style_loss_list, content_loss_list)
    return nnmodel

if __name__=="__main__":
    # print("Beginning Training Activity")
    trained_model = train_model()

    save_model_path = "/scratch/jv1589/Project/perceptual-losses/working_model.pth"
    torch.save(trained_model.state_dict(), save_model_path)
    trained_model = trained_model.eval()

    # trained_model = NNModel()
    # trained_model.load_state_dict(torch.load(save_model_path))
    # trained_model = trained_model.cuda()

    evaluate_style_transfer(trained_model, "/scratch/jv1589/Project/perceptual-losses/images/inputs/content/kinkaku_ji.jpg")
