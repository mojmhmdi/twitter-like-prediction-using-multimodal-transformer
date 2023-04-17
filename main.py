
import pandas as pd
import pandas as pd
import torch
import torch.nn as nn
from data_preperation import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model import *

data0 = pd.read_pickle('data_final.pkl')

device = torch.device("cuda")

# read and prepare the data

# model hyper parameters
max_seq_len = 100
batch_size = 1000
learning_rate = 3e-4
epochs = 100

def main():
    # train test split
    scaler_x, scaler_y, train_features, test_features, train_text, test_text, train_output, test_output = data_preprocessing(data0)

    # creat tokens and make everything a tensor
    train_text_seq, train_text_mask, test_text_seq, test_text_mask = tokenization_tensorization(max_seq_len,  train_text, test_text)


    # create dataloaders
    train_dataloader = DataLoader(TensorDataset(train_text_seq, train_text_mask, train_output, train_features), sampler=RandomSampler(TensorDataset(train_text_seq, train_text_mask, train_output, train_features)), batch_size=batch_size)
    # val_dataloader = DataLoader(TensorDataset(val_text_seq, val_text_mask, val_output, val_features), sampler=RandomSampler(TensorDataset(val_text_seq, val_text_mask, val_output, val_features)), batch_size=batch_size)
    test_dataloader = DataLoader(TensorDataset(test_text_seq, test_text_mask, test_output, test_features), sampler=RandomSampler(TensorDataset(test_text_seq, test_text_mask, test_output, test_features)), batch_size=batch_size)

    # load bert model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    bert.eval()
    for param in bert.parameters():
        param.requires_grad = False


    # define the model
    model = mmultimodal(bert,  features_input_size = train_features.shape[1])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # number of parameters in model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    # define parts of the model
    loss_function = nn.MSELoss()


    def train():
        
        model.train()
        total_loss = 0
        total_loss1 = 0
    
        for step,batch in enumerate(train_dataloader):
            print(step)
            if step % 10 == 0 and not step == 0:

                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
                print(total_loss1 / 10)
                total_loss1 = 0

            
            
            batch = [r.to(device) for r in batch]

            sent_id, mask, output, features = batch

            model.zero_grad()        

            preds = model(sent_id, mask, features)

            loss = loss_function(preds, output)


            loss.backward()
            # this may need to be commented
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            with torch.no_grad():

                total_loss = total_loss + loss.item()

                total_loss1 = total_loss1 + loss.item()

            #print(loss.item())

            # preds=preds.detach().cpu().numpy()

            # total_preds = np.append(preds, preds)

        avg_loss = total_loss / len(train_dataloader)

            # total_preds  = np.concatenate(total_preds, axis=0)

        return avg_loss


    best_valid_loss = float('inf')

    train_losses=[]
    valid_losses=[]

    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        
        train_loss = train()
        
        #valid_loss = evaluate()
        
        if train_loss < best_valid_loss:
            best_valid_loss = train_loss
            torch.save(model.state_dict(), 'weights/saved_weights.pt')
        
        train_losses.append(train_loss)
        #valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        #print(f'Validation Loss: {valid_loss:.3f}')


if __name__ == '__main__':
    main()