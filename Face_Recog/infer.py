
import torch
import numpy as np
from extract_face import preprocessing
from models import ConvAngular
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def generate_embeds(model, filename):
    face = preprocessing(filename)
    model = model.to('cuda').eval()
    
    face = transforms.ToTensor()(face)
    face = face.unsqueeze(0).to('cuda')
    embeds = model(face, return_embedding = True)
    norm_embeds = F.normalize(embeds).to('cpu').detach().numpy()
    return norm_embeds 

    
def cal_cosine(embeds_class, input_embed):
    dist_class = []
    input_embed_norm = input_embed/np.linalg.norm(input_embed)
    for i in range(len(embeds_class)):
        embeds_class_norm =  embeds_class[i] / np.linalg.norm(embeds_class[i], axis = 1, keepdims = True)
        dist_class.append(np.max(np.dot(embeds_class_norm, input_embed_norm.T)))
    index = np.argmax(dist_class)
    return index, dist_class[index]


if __name__ == "__main__":
    #   Load model
    model = ConvAngular(loss_type = 'arcface')
    model.load_state_dict(torch.load('models\\model_vgg16.pth'))

    # Load embedding space
    embeds_class = np.load('models\\DB.npy', allow_pickle = True).item()

    #   Predict 
    test_path = 'Datasets\\face-recognition-data\\testset\\Vinales.png'
    in_embeds = generate_embeds(model, test_path)
    index, dist = cal_cosine(embeds_class, in_embeds)


    img = mpimg.imread(test_path)
    plt.imshow(img)
    if dist > 0.9:
        plt.title(f"Hello {index}")
    else:
        plt.title("Stranger")
    plt.show()
