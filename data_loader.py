import json
import os.path as osp
import random

from torch.utils import data
from torchvision import transforms as T
from PIL import Image

def load_json(json_path):

    with open(json_path, 'r') as fp:
        data_list = json.load(fp)

    return data_list


class StepDataset(data.Dataset):
    """Dataset class for the Polyevore dataset."""

    def __init__(self, config, transform, transform_edge, mode='train'):
        """Initialize and preprocess the Polyevore dataset."""
        self.image_dir = config['TRAINING_CONFIG']['NAME']
        self.data_list = load_json(osp.join('annotation', f'{config["TRAINING_CONFIG"]["NAME"]}_{mode}_model.json'))
        self.transform = transform
        self.transform_edge = transform_edge
        self.mode = config['TRAINING_CONFIG']['MODE']
        self.seed = config['TRAINING_CONFIG']['SEED']

        random.seed(self.seed)
        random.shuffle(self.data_list)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        anchor = self.data_list[index]['anchor']
        edge = self.data_list[index]['edge']
        close_list = self.data_list[index]['close_list']
        far_list = self.data_list[index]['far_list']

        anchor_img = Image.open(osp.join(self.image_dir, anchor)).convert('RGB')
        edge_img = Image.open(osp.join(self.image_dir, edge)).convert('L')
        close_img = Image.open(osp.join(self.image_dir, random.sample(close_list, 1)[0])).convert('RGB')
        far_img = Image.open(osp.join(self.image_dir, random.sample(far_list, 1)[0])).convert('RGB')

        return self.transform(anchor_img), self.transform(close_img), self.transform(far_img), self.transform_edge(edge_img)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_data_loader(config, mode='train'):

    transform = list()
    transform.append(T.Resize((config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'])))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = T.Compose(transform)

    transform_edge = list()
    transform_edge.append(T.Resize((config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'])))
    transform_edge.append(T.ToTensor())
    transform_edge.append(T.Normalize(mean=(0.5,), std=(0.5,)))

    transform_edge = T.Compose(transform_edge)

    dataset = StepDataset(config, transform, transform_edge, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(mode == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)

    return data_loader