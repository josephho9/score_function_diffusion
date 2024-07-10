import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

# Resize images to 14x14
x_train_small = tf.image.resize(x_train, (14, 14)).numpy()
x_test_small = tf.image.resize(x_test, (14, 14)).numpy()

# Reshape and normalize
x_train_small = x_train_small.reshape(60000, 196).astype('float32') / 255
x_test_small = x_test_small.reshape(10000, 196).astype('float32') / 255

train_size = 5000
dataset = x_train_small[:train_size]

def normalized_PDF(data, x, t):
    var_t = 1 - np.exp(-2 * t)
    cal1 = np.inner(x - np.exp(-t) * data, x - np.exp(-t) * data)
    cal2 = cal1 / (-2 * var_t)
    return np.exp(cal2)

def score(dataset, x, t):
    div = 0
    di = 0
    for data in dataset:
        nor_data = normalized_PDF(data, x, t)
        div += nor_data * (-1 * (1 - np.exp(-2 * t)))
        di += np.inner(nor_data, x - np.exp(-t) * data)
    return di / div

def predict(diff):
    return (diff < 0.02).astype(int)

def denoise_step(R_new, epo, t, Delta, R, pool):
    results = pool.starmap(score, [(dataset, R[-1][i], (epo - t) / Delta) for i in range(len(R_new))])
    for i in range(len(R_new)): 
        R_new[i] = R[-1][i] + (R[-1][i] + results[i]) / Delta
    return R_new

def main():
    Delta = 500
    epo = 100

    T = epo / Delta
    C = 0
    W = 0

    for k in tqdm(range(100)):
        F = [x_train_small[k]]
        for t in range(epo):
            F_new = [0] * 196
            for i in range(len(F_new)):
                x = np.random.normal(loc=0, scale=1 / Delta)
                F_new[i] = F[-1][i] - F[-1][i] / Delta + np.sqrt(2) * x
            F.append(F_new)

        R = [F[-1]]
        for t in tqdm(range(epo - 1)):
            print('t :', t)
            pool = ThreadPool(8)
            R_new = [0] * 196
            R_new = denoise_step(R_new, epo, t, Delta, R, pool)
            pool.close()
            pool.join()
            R.append(R_new)

        Diff = [F[0] - R[-1]]
        if np.inner(Diff, Diff) > 0.02:
            if k < train_size:
                W += 1
            else:
                C += 1
        else:
            if k < train_size:
                C += 1
            else:
                W += 1
    print(C)
    print(W)
    print(C / (C + W))

if __name__ == '__main__':
    main()
