
import os
import json
from PIL import Image
import numpy as np
import torch
import torch.utils.data as Data
from torchvision import transforms

import transformers
from transformers import BertModel, BertTokenizer

from consts import global_consts as gc


# only loads dataset with text and vision modality
class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.text = np.empty(0)
        self.text_info = np.empty(0)
        self.vision = np.empty(0)
        self.raw_vision_path = np.empty(0)
        self.y = np.empty(0)

class MultimodalDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train"):
        self.root = root
        self.cls = cls
        if len(MultimodalDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MultimodalDataset.trainset
        elif self.cls == "test":
            self.dataset = MultimodalDataset.testset
        elif self.cls == "valid":
            self.dataset = MultimodalDataset.validset

        self.text = self.dataset.text
        self.text_info = self.dataset.text_info
        self.vision = self.dataset.vision
        self.raw_vision_path = self.dataset.raw_vision_path
        self.y = self.dataset.y


    def load_data(self):
        dataset_path = os.path.join(gc.data_path, gc.dataset + '/')
        loaded_data = []
        if self.cls == "train":
            cls_string = "train"
        elif self.cls == "valid":
            cls_string = "dev_seen"
        elif self.cls == "test":
            cls_string = "test_seen"
        else:
            raise Exception("No cls_string set!")
        print("Loading {} dataset...".format(self.cls))
        with open(dataset_path+cls_string+'.jsonl','r') as f:
            for json_string in list(f):
                loaded_data.append(json.loads(json_string))
                # loaded data example(8500 datapoints laoded):
                # {
                # 'id': '01649',
                # 'img': 'img/01649.png',
                # 'label': 1,
                # 'text': 'american cops when they graduate police academy let the black man come forth! let murder be done upon him!'
                # }

        img_list = [x["img"] for x in loaded_data]
        imgs = [img for img in img_list if img not in gc.filter_image_list]
        labels = [x["label"] for x in loaded_data if x["img"] not in gc.filter_image_list]
        texts = [str(x["text"]) for x in loaded_data if x["img"] not in gc.filter_image_list]

        # should have 8435 datapoints after filtering out images with dimension < 224
        # text encoder
        encoded_texts = []
        pretrain_model = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(pretrain_model)
        # bert_model = BertModel.from_pretrained(pretrain_model)
        max_len =160
        for t in texts:
            encoded_text = tokenizer.encode_plus(
                                  t,
                                  add_special_tokens=True,
                                  max_length=max_len,
                                  return_token_type_ids=False,
                                  pad_to_max_length=True,
                                  # padding='max_length',
                                  return_attention_mask=True,
                                  return_tensors='np', # "pt" for pytorch tensors
                                )
            encoded_texts += [encoded_text]
        print(texts[0], encoded_texts[0])
        '''
        encoded_texts for "its their character not their color that matters":
        {'input_ids': [101, 1157, 1147, 1959, 1136, 1147, 2942, 1115, 5218, 102,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0],
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        '''

        # image encoder
        encoded_imgs = []
        imagenet_transform = transforms.Compose([#transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        for img in imgs:
            image_folderpath = os.path.join(gc.data_path, gc.dataset + '/')
            encoded_img = imagenet_transform(Image.open(image_folderpath+img))
            encoded_imgs += [encoded_img.cpu().detach().numpy()]
        '''
        img/42953.png encoded example:
        [[[-2.117904   -2.117904   -2.117904   ... -2.117904   -2.117904
           -2.117904  ]
          [-2.117904   -2.117904   -2.117904   ... -2.117904   -2.117904
           -2.117904  ]
          [ 0.05693974  0.05693974  0.05693974 ...  0.05693974  0.05693974
            0.05693974]
          ...
          [-2.1007793  -2.1007793  -2.1007793  ... -1.0390445  -1.0561693
           -1.0219197 ]
          [-2.117904   -2.117904   -2.117904   ... -2.117904   -2.117904
           -2.117904  ]
          [-2.117904   -2.117904   -2.117904   ... -2.117904   -2.117904
           -2.117904  ]]

         [[-2.0357141  -2.0357141  -2.0357141  ... -2.0357141  -2.0357141
           -2.0357141 ]
          [-2.0357141  -2.0357141  -2.0357141  ... -2.0357141  -2.0357141
           -2.0357141 ]
          [ 0.18767506  0.18767506  0.18767506 ...  0.18767506  0.18767506
            0.18767506]
          ...
          [-2.0182073  -2.0182073  -2.0182073  ... -1.3879551  -1.4229691
           -1.370448  ]
          [-2.0357141  -2.0357141  -2.0357141  ... -2.0357141  -2.0357141
           -2.0357141 ]
          [-2.0357141  -2.0357141  -2.0357141  ... -2.0357141  -2.0357141
           -2.0357141 ]]

         [[-1.8044444  -1.8044444  -1.8044444  ... -1.8044444  -1.8044444
           -1.8044444 ]
          [-1.8044444  -1.8044444  -1.8044444  ... -1.8044444  -1.8044444
           -1.8044444 ]
          [ 0.40906325  0.40906325  0.40906325 ...  0.40906325  0.40906325
            0.40906325]
      ...
      [-1.7870152  -1.7870152  -1.7870152  ... -1.3512855  -1.3687146
       -1.3338562 ]
      [-1.8044444  -1.8044444  -1.8044444  ... -1.8044444  -1.8044444
       -1.8044444 ]
      [-1.8044444  -1.8044444  -1.8044444  ... -1.8044444  -1.8044444
       -1.8044444 ]]]

        '''

        self.text = texts
        self.raw_vision_path = imgs
        self.text_info = encoded_texts
        self.vision = np.array(encoded_imgs)
        self.y = np.array(labels)
        print(self.text[0], self.vision[0], self.y[0])


        print("Dimension of {} {} is {}.".format(self.cls, "raw text set", len(self.text)))
        print("Dimension of {} {} is {}.".format(self.cls, "vision set", self.vision.shape))
        print("Dimension of {} {} is {}.".format(self.cls, "labels", self.y.shape))
        # Dimension of train raw text set is 8435.
        # Dimension of train vision set is (8435, 3, 224, 224).
        # Dimension of train labels is (8435,).


    def __getitem__(self, index):
        inputLen = len(self.text[index])
        return self.text[index], self.vision[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MultimodalDataset(gc.data_path)
