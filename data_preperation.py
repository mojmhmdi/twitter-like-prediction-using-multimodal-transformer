
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
import torch

def remove_usernames_links(tweet):
    # tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    return tweet


def data_maker(data0):
    data0.index = range(data0.shape[0])
    
    # filter reply tweets
    index = data0['Text'].str.startswith('@', na=False)
    data0 = data0[index == False]
    # filter tweets starting with links
    index = data0['Text'].str.startswith('http', na=False)
    data0 = data0[index == False]
    data0.index = range(data0.shape[0])
    data0['Text'] = data0['Text'].apply(remove_usernames_links)

    # data of the owner of the account
    data_account_owner = data0.groupby(['user.username']).sample()
    data_account_owner.index = range(data_account_owner.shape[0])
    data_account_owner = data_account_owner.drop(columns=['date', 'Text', 'viewCount', 'likeCount', 'replyCount',
        'retweetCount', 'user.id','user.mediaCount', 'user.listedCount','user.statusesCount', 'lang', 'hashtag',
        'user.displayname', 'user.renderedDescription','user.descriptionLinks','user.location'])
    data_mean = data0.loc[:, ['user.username', 'likeCount', 'replyCount', 'user.statusesCount', 'user.mediaCount', 'user.listedCount',
                                    'retweetCount','viewCount','user.created']].groupby(['user.username']).mean()
    data_size = (data0.groupby(['user.username']).size())


    data_account_owner.loc[:,['likemean']] = list(data_mean ['likeCount'][:])
    data_account_owner.loc[:,['viewmean']] = list(data_mean ['viewCount'][:])
    data_account_owner.loc[:,['replymean']] = list(data_mean ['replyCount'][:])
    data_account_owner.loc[:,['statusesmean']] = list(data_mean ['user.statusesCount'][:])
    data_account_owner.loc[:,['listedmean']] = list(data_mean ['user.listedCount'][:])
    data_account_owner.loc[:,['mediamean']] = list(data_mean ['user.mediaCount'][:])
    data_account_owner.loc[:,['retweetmean']] = list(data_mean ['retweetCount'][:])
    data_account_owner.loc[:,['tweetscount']] = data_size.values
    data_account_owner.loc[:,['user.created']] = [k.year for k in data_account_owner['user.created']]

    data0 = data0.loc[:,['date','Text', 'user.username', 'likeCount', 'replyCount', 'retweetCount','lang']]
    data0.loc[:,['date']] = [k.weekday() for k in data0['date']]

    data0.loc[:,['hour']] = 0
    data0.loc[:,['hour']] = [k.hour for k in data0['date']]

    for i in data_account_owner.columns [1:]:
        data0[i] = 0
        
    for i in range(data0.shape[0]):
        
        # data0['Text'][i] = tweet_cleaning (data0['Text'][i])

        if (a := data_account_owner[data_account_owner['user.username']==data0['user.username'][i]]).shape[0] != 0:
            data0.iloc[i,8:] = a.iloc[0,1:]
            if i%1000 ==0:
                print(i)

    data0 ['user.username']
    data0 = data0[data0['lang'] == 'en']
    data0.index = range(data0.shape[0])
    data_account_owner.index = range(data_account_owner.shape[0])

    return data0, data_account_owner

def data_preprocessing(data0):
    data0.loc[:,['date']] = [k.weekday() for k in data0['date']]

    # data0 = data0.drop(columns = ['lang','user.username','user.verified','replyCount', 'retweetCount','user.followersCount','user.created','viewmean','statusesmean', 'listedmean', 'mediamean', 'retweetmean'])
    data0 = data0.drop(columns = ['lang','user.username','user.verified','replyCount', 'retweetCount','viewmean','statusesmean', 'listedmean', 'mediamean','user.created','user.friendsCount'])

    output_data = data0['likeCount'].tolist()
    input_features = data0.drop(columns = ['Text','likeCount'])
    input_text = data0['Text'].tolist()
    del data0
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaler_x = scaler_x.fit(input_features)
    input_features = scaler_x.transform(input_features)

    scaler_y = scaler_y.fit(np.array(output_data).reshape(-1,1))
    output_data = scaler_y.transform(np.array(output_data).reshape(-1,1))

    # input_features = np.concatenate((input_features, np.array([input_text]).reshape(-1,1)), axis = -1)

    # val_features, test_features, val_output, test_output = train_test_split(temp_features, temp_output, random_state=2018, test_size=0.5)


    # train test validation split
    train_features, test_features, train_text, test_text, train_output, test_output = train_test_split(input_features, np.array([input_text]).reshape(-1,1), output_data, random_state=2018, test_size=0.2)
    del input_features, output_data

    # val_features, test_features, val_output, test_output = train_test_split(temp_features, temp_output, random_state=2018, test_size=0.5)
    return scaler_x, scaler_y, torch.tensor(np.array(train_features, dtype = np.float32)), torch.tensor(np.array(test_features, dtype = np.float32)), train_text, test_text, torch.tensor(np.array(train_output, dtype = np.float32)), torch.tensor(np.array(test_output, dtype = np.float32))
    # return scaler_x, scaler_y, np.array(train_features[:,:-1], dtype = np.float32), np.array(train_output, dtype = np.float32), np.array(val_features[:,:-1], dtype = np.float32), np.array(val_output, dtype = np.float32), np.array(test_features[:,:-1], dtype = np.float32), np.array(test_output, dtype = np.float32), train_features[:,-1].tolist(), val_features[:,-1].tolist(), test_features[:,-1].tolist()

def tokenization_tensorization(max_seq_len, train_text, test_text):
    """this function converts the text data into tokens uing BERT pretrained model and then makes everything a tensor"""

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens_train = tokenizer.batch_encode_plus(
        train_text[:,0].tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )



    tokens_test = tokenizer.batch_encode_plus(
        test_text[:,0].tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )


    train_text_seq = torch.tensor(tokens_train['input_ids'])
    train_text_mask = torch.tensor(tokens_train['attention_mask'])

    test_text_seq = torch.tensor(tokens_test['input_ids'])
    test_text_mask = torch.tensor(tokens_test['attention_mask'])

    
    return train_text_seq, train_text_mask, test_text_seq, test_text_mask
