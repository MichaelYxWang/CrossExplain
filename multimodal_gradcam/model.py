import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import transformers
from transformers import BertModel
from consts import global_consts as gc

class MultimodalClassifier(nn.Module):
    def __init__(self, n_classes=gc.class_num):
        super(MultimodalClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(gc.text_pretrained_model)
        self.vgg = vgg19(pretrained=True)
        self.features_conv = self.vgg.features[:36]
        self.features_final = self.vgg.classifier[:-1]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.drop = nn.Dropout(p=gc.dropout)
        self.combined_fc1 = nn.Linear(self.bert.config.hidden_size + 4096, 256)
        self.combined_fc2 = nn.Linear(256, n_classes)

    def forward(self, x, input_ids, attention_mask):
        x = self.features_conv(x)
        x = self.max_pool(x)
        x = x.view((x.size(0), -1))
        x = self.features_final(x)

        last_hidden_state, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          return_dict=False
        )

        combined = torch.cat((x, pooled_output), 1)
        combined = F.relu(self.combined_fc1(combined))
        combined = F.relu(self.combined_fc2(combined))

        return combined
