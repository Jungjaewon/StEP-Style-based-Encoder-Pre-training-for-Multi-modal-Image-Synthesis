import json
import cv2
import os.path as osp
import glob
import torch
import numpy as np
import random

from tqdm import tqdm
from bit_models import KNOWN_MODELS
from numpy import dot
from numpy.linalg import norm
from torchvision import transforms as T
from PIL import Image

transform = list()
transform.append(T.Resize((256, 256)))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
transform = T.Compose(transform)


def vec_normalize(vector):
    norm = np.linalg.norm(vector)
    normed_vector = vector / norm
    return normed_vector


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def load_json(json_path):
    with open(json_path, 'r') as fp:
        data_list =  json.load(fp)
    return data_list


def get_tensor(img_path):

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def get_edge_image(img_dir):

    for img_path in tqdm(glob.glob(osp.join(img_dir, '*.jpg'))):
        img_name = osp.basename(img_path)
        edge_name = img_name.replace('.jpg', '_edge.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        edge_img = cv2.Canny(img, 50, 200)
        cv2.imwrite(osp.join(img_dir, edge_name), 255 - edge_img)


def bit_model():
    model = KNOWN_MODELS['BiT-M-R101x1'](head_size=21843, zero_head=True)
    model.load_from(np.load('BIT-M-R101x1.npz'))
    model = model.to('cuda:0').eval()
    x = torch.rand((1, 3, 256, 256)).to('cuda:0')
    print(model(x).size())


def get_item_db(model, name, img_list, device='cuda:0'):

    img_db = list()
    img_list = [x for x in img_list if 'edge' not in x]

    for img_path in tqdm(img_list):
        img_path = osp.join(name, img_path)
        img_tensor = get_tensor(img_path).to(device)
        feature = model(img_tensor).detach().cpu().numpy()
        feature = vec_normalize(feature)
        img_db.append([img_path, feature])

    return img_db


def build_dataset_json(mode, name, device='cuda:0', seed=2355):

    result_data_list = list()
    random.seed(seed)

    model = KNOWN_MODELS['BiT-M-R101x1'](head_size=21843, zero_head=True)
    model.load_from(np.load('BiT-M-R101x1.npz'))
    model = model.to(device).eval()

    img_list = load_json(osp.join('annotation', f'{name}_{mode}.json'))
    img_db = get_item_db(model, name, img_list)

    for img_name in img_list:

        search_result = list()
        img_path = osp.join(name, img_name)
        img_tensor = get_tensor(img_path).to(device)
        base_feature = model(img_tensor).detach().cpu().numpy()
        base_feature = vec_normalize(base_feature)

        random.shuffle(img_db)

        for db_img_path, feature in img_db[:3000]:
            dist = float(cos_sim(np.squeeze(base_feature),np.squeeze(feature)))
            search_result.append([dist, db_img_path])

        search_result = np.array(search_result)
        idx = np.argsort(search_result, axis=0)[::-1]
        search_result = search_result[idx[:, 0]]

        #print(search_result[:10])
        #assert search_result[0,0] == img_path

        closet_list = list(search_result[1:27, 1])[6:]
        farset_list = list(search_result[-21:, 1])

        #print(f'img_path : {img_path}')
        #print(f'closet_list : {closet_list}')
        #print(f'farset_list : {farset_list}')

        closet_list = [osp.basename(x) for x in closet_list]
        farset_list = [osp.basename(x) for x in farset_list]
        result_data_list.append({'anchor': img_name, 'close_list': closet_list, 'far_list': farset_list,
                                 'edge': img_name.replace('.jpg','_edge.jpg')})

    with open(osp.join('annotation', f'{name}_{mode}_model.json'), 'w') as fp:
        json.dump(result_data_list, fp)


def data_split(img_dir, seed=111):
    # this function for building data for metric learning.
    random.seed(seed)
    img_list = glob.glob(osp.join(img_dir, '*.jpg'))
    random.shuffle(img_list)
    train_cut = int(0.8 * len(img_list))

    test_val_cut = 0.5

    train_data = img_list[:train_cut]
    val_test_data = img_list[train_cut:]
    val_data = val_test_data[:int(len(val_test_data) * test_val_cut)]
    test_data = val_test_data[int(len(val_test_data) * test_val_cut):]

    for mode, data_list in [['train', train_data], ['val', val_data], ['test', test_data]]:
        result_list = [osp.basename(x) for x in data_list]
        with open(osp.join('annotation', f'{img_dir}_{mode}.json'),'w') as fp:
            json.dump(result_list, fp)


if __name__ == '__main__':
    pass
    #bit_model()

    """
    # build json
    # bags
    data_split('bags', seed=111)
    build_dataset_json('train', 'bags', device='cuda:0')
    build_dataset_json('val', 'bags', device='cuda:0')
    build_dataset_json('test', 'bags', device='cuda:0')

    # tops
    data_split('tops', seed=111)
    build_dataset_json('train', 'tops', device='cuda:0')
    build_dataset_json('val', 'tops', device='cuda:0')
    build_dataset_json('test', 'tops', device='cuda:0')
    """

    # make edge image
    get_edge_image('bags')
    get_edge_image('tops')
