
import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#from data_preperation import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoModel, BertTokenizerFast


# load bert model
bert = AutoModel.from_pretrained('bert-base-uncased')
bert.eval()
for param in bert.parameters():
    param.requires_grad = False



class mmultimodal(nn.Module):
    
    def __init__ (self, bert, Droupout_rate = 0.2, features_input_size = 12, hidden_size_mode1 = 512, output_size_mode1 = 128, hidden_size_mode2 = 512, output_size_mode2 = 20):
        
        super(mmultimodal, self).__init__()

        self.bert = bert

        self.dropout = nn.Dropout(Droupout_rate)
        
        self.relu =  nn.ReLU()

        self.mode1_fc1 = nn.Linear(768,hidden_size_mode1)
        
        self.mode1_fc2 = nn.Linear(hidden_size_mode1,output_size_mode1)
        
        self.mode1_fc3 = nn.Linear(hidden_size_mode1,hidden_size_mode1)
        
        self.mode1_fc4 = nn.Linear(hidden_size_mode1,hidden_size_mode1)

        self.mode2_fc1 = nn.Linear(features_input_size, hidden_size_mode2)

        self.mode2_fc2 = nn.Linear(hidden_size_mode2, output_size_mode2)
        
        self.mode2_fc3 = nn.Linear(hidden_size_mode2,hidden_size_mode2)
        
        self.mode2_fc4 = nn.Linear(hidden_size_mode2,hidden_size_mode2)

        self.output_net_fc1 = nn.Linear(output_size_mode1+output_size_mode2, 128)
        
        self.output_net_fc3 = nn.Linear(128, 64)

        self.output_net_fc2 = nn.Linear(64, 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, sent_id, mask_id, input_features):
        
        bert_embedding = bert(sent_id, attention_mask=mask_id).last_hidden_state 
        
        ber_embedding = torch.mean(bert_embedding, axis = 1)
        
        x1 = self.relu(self.mode1_fc1(ber_embedding))
        # we may need to change the location of this layer here
        x1 = self.relu( self.mode1_fc3(x1))
        
        x1 = self.dropout(x1)
        
        x1 = self.relu( self.mode1_fc4(x1))

        # we may need a relu here
        x1 = self.relu(self.mode1_fc2(x1))
        
        x2 = self.relu(self.mode2_fc1(input_features))
        
        x2 = self.relu( self.mode2_fc3(x2))
        
        x2 = self.relu( self.mode2_fc4(x2))

        # we may need to change this layer
        x2 = self.dropout(x2)

        x2 = self.relu(self.mode2_fc2(x2))
        
        concatenated_output = torch.cat((x1,x2), dim=1) # Shape: (batch_size, bert_output_dim + fc_float_output_dim)

        output_tensor = self.relu(self.output_net_fc1(concatenated_output))

        output_tensor = self.relu(self.output_net_fc3(output_tensor))

        output_tensor = self.output_net_fc2(output_tensor)
        
        
        return output_tensor