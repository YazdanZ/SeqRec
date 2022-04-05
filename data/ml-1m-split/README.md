## Data Format
The data format is the same as in the movielens data, just in HDF5 format in pandas dataframes with indices. Furthermore, it is split into train, test, and validation sets (80-10-10) on a user basis.

## Data Organization
Each user .h5 contains 3 dataframes, accessible by keys:
 - **users**: user metadata
 - **users_history**: past ratings for each user
 - **samples_future**: future ratings for each user. *This is the set we do training and inference on.*

The movies .h5 is simply the movies information repackaged.

## Loading the data
To load the data, simply use:

```pd.read_hdf(os.path.join(save_dir,'users_train_dfs.h5'), key='users_history')```

replacing key and the directory as needed.

For the movies dataframe, a key is unnecessary.

