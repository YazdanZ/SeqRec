
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MovielensDataset(torch.utils.data.Dataset):
    """Movielens Dataset - Returns a User/Movie pair with rating."""

    def __init__(self, users_file, movies_file, user_history=False, user_info=False, movie_info=False, lengths=False, pad='pre', max_history_length=0):
        """
        Args:
            users_file (string): Path to the .h5 file with user data.
            movies_file (string): Path to the .h5 file with movies data.
            user_history (bool, optional): Use user's sequential ratings history.
            user_info (bool, optional): Use user's metadata.
            movie_info (bool, optional): Use movie data.
        """
        self.lengths=lengths
        if user_history:
            # load user histories
            self.users_history_df = pd.read_hdf(users_file, key='users_history')
            self.users_history_df = self.users_history_df.droplevel(0)
            # self.users_history_df = self.users_history_df[self.users_history_df['rating'] >= 4]
            count = self.users_history_df.groupby('user_id').count()
            self.max_seq_len = count.max().max()
            # self.min_seq_len = self.users_history_df.groupby('user_id').count().min().min()
            # print(self.max_seq_len, self.min_seq_len)

            # sort by timestamp
            self.users_history_df = self.users_history_df.groupby('user_id').apply(lambda x: x.sort_values('timestamp'))
            self.users_history_df = self.users_history_df.drop('timestamp', axis=1)
            self.users_history_df = self.users_history_df.droplevel('user_id')

            # turn into sequences and pad
            self.users_history_df = self.users_history_df.groupby('user_id').agg(lambda x: x.tolist()[-max_history_length:])

            pad_to = min(self.max_seq_len, max_history_length) if max_history_length > 0 else self.max_seq_len
            self.users_history_df = self.users_history_df.applymap(np.asarray, dtype=int)#.iloc[:,0]

            
            if lengths:
                self.users_history_df['lengths_cut'] = self.users_history_df.movie_id.map(len).astype(np.int64)
                self.users_history_df['lengths'] = count.movie_id.astype(np.int64)
                
            if pad == 'pre':
                self.users_history_df.movie_id = self.users_history_df.movie_id.map(
                    lambda x: np.pad(x, (pad_to - len(x), 0), 'constant', constant_values=(0,0)))
                self.users_history_df.rating = self.users_history_df.rating.map(
                    lambda x: np.pad(x, (pad_to - len(x), 0), 'constant', constant_values=(3,3)))
            elif pad == 'post':
                self.users_history_df.movie_id = self.users_history_df.movie_id.map(
                    lambda x: np.pad(x, (0, pad_to - len(x)), 'constant', constant_values=(0,0)))
                self.users_history_df.rating = self.users_history_df.rating.map(
                    lambda x: np.pad(x, (0, pad_to - len(x)), 'constant', constant_values=(3,3)))
        else:
            self.users_history_df = None

        # TODO: Preprocessing user_info
        
        self.users_df = pd.read_hdf(users_file, key='users')
        self.users_df = self.users_df.drop('zip_code', axis=1)
        # self.users_df.gender = self.users_df.gender.map({'M':0.0, 'F':1.0})
        self.users_df.age = self.users_df.age.map(lambda x: x / 56.0)

        # one-hot occupation and gender
        categories = [i for i in range(21)]
        enc  = OneHotEncoder(categories=[categories], sparse=False)
        occupations = enc.fit_transform(self.users_df.occupation.to_numpy().reshape(-1, 1))
        names = [f'occupation: {name}' for name in categories]
        occupations_df = pd.DataFrame(occupations, index=self.users_df.index, columns = names, dtype=np.float32)

        gender_cats = ['M', 'F']
        enc  = OneHotEncoder(categories=[gender_cats], sparse=False)
        genders = enc.fit_transform(self.users_df.gender.to_numpy().reshape(-1, 1))
        names = [f'gender: {name}' for name in gender_cats]
        gender_df = pd.DataFrame(genders, index=self.users_df.index, columns = names, dtype=np.float32)
        
        self.users_df = pd.concat([self.users_df.drop(['occupation', 'gender'], axis=1), gender_df, occupations_df], axis=1).astype(np.float32)
        self.user_info = user_info


        # TODO: Preprocessing movie_info
        self.movie_info = movie_info
        movies_df = pd.read_hdf(movies_file)
        self.movies_genres_df = movies_df['genres']
        self.movies_title_df = movies_df[('title', 'title')]

        
        model = SentenceTransformer('all-MiniLM-L6-v2')

        #Sentences are encoded by calling model.encode()
        self.movies_title_df = pd.DataFrame(model.encode(self.movies_title_df.to_numpy()), index=self.movies_title_df.index)

        self.samples_future_df = pd.read_hdf(users_file, key='samples_future')



    def __len__(self):
        return len(self.samples_future_df)

    def __getitem__(self, idx):
        user_id, movie_id, rating, timestamp = tuple(self.samples_future_df.iloc[idx])

        sample = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp
        }
        

        if self.users_history_df is not None:
            sample['user_history_rating'] = self.users_history_df.loc[user_id, 'rating']
            sample['user_history_movie_id'] = self.users_history_df.loc[user_id, 'movie_id']
            if self.lengths:
                sample['user_history_len_cut'] = self.users_history_df.loc[user_id, 'lengths_cut']
                sample['user_history_len'] = self.users_history_df.loc[user_id, 'lengths']
 
        # TODO: Output user_info
        if self.user_info:
            sample['user_info'] = self.users_df.loc[user_id].to_numpy()

        # TODO: Output movie_info
        if self.movie_info:
            sample['movie_info_genres'] = self.movies_genres_df.loc[movie_id].to_numpy()
            sample['movie_info_title'] = self.movies_title_df.loc[movie_id].to_numpy()

        return sample