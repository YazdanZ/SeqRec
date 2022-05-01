# SeqRec

## Abstract
**Motivation:** Recommendation systems are used by major services like YouTube, Netflix and Amazon to
recommend items or content of interest to their users. Sequential patterns in data play a significant role
in how well a recommendation system performs. However, the struggle to uncover complex sequential
relationships in a user’s history is still common.


**Results:** In this study, we propose a deep learning based approach that utilizes the user’s history. Using
a multiplicative long short term memory (mLSTM), we capture the sequential information of a user. On
the MovieLens 1M dataset of 6040 users and 3706 movies, resulting in over a million interactions, we
trained a deep learning network to capture the sequential information of each user by utilizing the user
characteristics, the item features and the user-item history. Our highest reported Mean Squared Error
(MSE) was 0.958.

[Project Report](https://github.com/YazdanZ/SeqRec/blob/master/ESCE_552_Project_Report__Group_10.pdf)
