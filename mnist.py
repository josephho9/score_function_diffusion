import numpy as np  # advanced math library
import random  # for generating random numbers
import tensorflow as tf
from tqdm import tqdm
import torch
import os
import ssl
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ssl._create_default_https_context = ssl._create_unverified_context


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]


x_train_small = torch.from_numpy(tf.image.resize(x_train, (14,14)).numpy()).float().to('cuda')
x_test_small = torch.from_numpy(tf.image.resize(x_test, (14,14)).numpy()).float().to('cuda')


x_train_small = x_train_small.reshape(60000, 196) # reshape 60,000 14 x 14 matrices into 60,000 196-length vectors.
x_test_small = x_test_small.reshape(10000, 196)   # reshape 10,000 14 x 14 matrices into 10,000 196-length vectors.

flattened_size = 14*14


x_train_small /= 255  # normalize each value for each pixel for the entire vector for each input
x_test_small /= 255

sample_size = 5
train_eval = random.choices(x_train_small, k=sample_size)
test_eval = random.choices(x_test_small, k=sample_size)

eval = [{'label': 'train', 'data': data} for data in train_eval] + [{'label': 'test', 'data': data} for data in test_eval]

random.shuffle(eval)

total = len(eval)

def normalized_PDF(x, t):
    var_t = 1 - torch.exp(-2 * t)
    cal1 = torch.einsum('ij,ij->i', x - torch.exp(-t) * x_train_small, x - torch.exp(-t) * x_train_small)
    cal2 = cal1 / (-2 * var_t)
    return torch.exp(cal2)


def score(x, t):
    x = torch.tensor(x).to('cuda')
    gpu_nor_data = normalized_PDF(x, t)
    div = gpu_nor_data.sum() * (-1 * (1 - torch.exp(-2 * t)))
    di = torch.einsum('i,ij->j', gpu_nor_data, (x - torch.exp(-t) * x_train_small))
    return di / div


Delta = 500
epo = 50

T = epo / Delta
C = 0
W = 0
test_cum_sim = 0
train_cum_sim = 0

for k in tqdm(range(total)):
    entry = eval[k]
    F = [entry['data']]
    label = entry['label']

    # Forward Process
    for t in range(epo):
        F_new = [0] * flattened_size
        for i in range(len(F_new)):
            x = torch.normal(mean=torch.tensor(0.0).float(), std=torch.tensor(1.0 / Delta).float())
            F_new[i] = F[-1][i] - F[-1][i] / Delta + torch.sqrt(torch.tensor(2.0).float()) * x
        F.append(F_new)

        # Save tensor every 10 steps
        if t % 10 == 0:
            torch.save(torch.tensor(F_new).to('cuda'), f'noising_step_{t}_sample_{k}.pt')

    R = [F[-1]]

    # Reverse Process
    for t in tqdm(range(epo - 1)):
        t_tensor = torch.tensor(t).to('cuda')
        R_new = torch.zeros(flattened_size).to('cuda')
        for i in range(len(R_new)):
            R_new[i] = R[-1][i] + (R[-1][i] + score(R[-1], (epo - t_tensor) / Delta)[i]) / Delta
        R.append(R_new)

        # Save tensor every 10 steps
        if t % 10 == 0:
            torch.save(torch.tensor(R_new).to('cuda'), f'denoising_step_{t}_sample_{k}.pt')

    # Calculate difference between original image (F[0]) and final score-guided denoised step (R_new)
    Diff = torch.tensor(F[0]).to('cuda') - torch.tensor(R[-1]).to('cuda')

    sim = torch.inner(Diff, Diff).item()

    print(sim)
    if label == 'test':
        test_cum_sim += sim
        if sim > 0.0075:
            C += 1
            print('c')

    elif label == 'train':
        train_cum_sim += sim
        if sim <= 0.0075:
            C += 1
            print('c')

print(C)
print(C / total)
print(train_cum_sim / sample_size)
print(test_cum_sim / sample_size)
