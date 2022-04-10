
import torch
import numpy as np
import pandas as pd

class MovielensDataset(torch.utils.data.Dataset):
    """Movielens Dataset - Returns a User/Movie pair with rating."""

    def __init__(self, users_file, movies_file, user_history=False, user_info=False, movie_info=False, max_history_length=0):
        """
        Args:
            users_file (string): Path to the .h5 file with user data.
            movies_file (string): Path to the .h5 file with movies data.
            user_history (bool, optional): Use user's sequential ratings history.
            user_info (bool, optional): Use user's metadata.
            movie_info (bool, optional): Use movie data.
        """
        if user_history:
            # load user histories
            self.users_history_df = pd.read_hdf(users_file, key='users_history')
            self.users_history_df = self.users_history_df.droplevel(0)
            # self.users_history_df = self.users_history_df[self.users_history_df['rating'] >= 4]
            self.max_seq_len = self.users_history_df.groupby('user_id').count().max().max()
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
            self.users_history_df = self.users_history_df.applymap(
                lambda x: np.pad(x, (pad_to - len(x), 0), 'constant', constant_values=(0,0)))
        else:
            self.users_history_df = None

        # TODO: Preprocessing user_info
        self.user_info = user_info
        self.users_df = pd.read_hdf(users_file, key='users')


        # TODO: Preprocessing movie_info
        self.movie_info = movie_info
        self.movies_df = pd.read_hdf(movies_file)

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
 
        # TODO: Output user_info
        if self.user_info:
            sample['user_info'] = self.users_df.loc[user_id].to_dict()

        # TODO: Output movie_info
        if self.movie_info:
            sample['movie_info'] = self.movies_df.loc[movie_id].to_dict()

        return sample