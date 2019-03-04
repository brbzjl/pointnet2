def sample_data_label(data, label, num_sample):
    datas = []
    labels = []
    N = data.shape[0]-1
    idx = np.arange(N)
    np.random.shuffle(idx)
    if N >= num_sample:
        num = int(np.round(N / num_sample))
        for num in range(num):
            start_idx = num * num_sample
            end_idx = (num + 1) * num_sample
            if end_idx <= N:
                datas.append(np.expand_dims(data[idx[start_idx:end_idx]], 0))
                labels.append(np.expand_dims(label[idx[start_idx:end_idx]], 0))
            else:
                data_temp = data[idx[0:end_idx - N +1]]
                label_temp = label[idx[0:end_idx - N+1]]
                data_temp = np.concatenate([data[idx[start_idx:-1]], data_temp], axis=0)
                label_temp = np.concatenate([label[idx[start_idx:-1]], label_temp], axis=0)
                datas.append(np.expand_dims(data_temp, 0))
                labels.append(np.expand_dims(label_temp, 0))
    else:
        num = int(np.round(num_sample / N))
        for num in range(num):
            start_idx = num * N
            end_idx = (num + 1) * N
            if end_idx <= num_sample-1:
                datas.append(np.expand_dims(data), 0)
                labels.append(np.expand_dims(label), 0)
            else:
                data_temp = data[:, num_sample - start_idx+2]
                label_temp = label[:, num_sample - start_idx+2]
                datas.append(np.expand_dims(data_temp), 0)
                labels.append(np.expand_dims(label_temp)  , 0)

    return np.concatenate(datas, 0), np.concatenate(labels, 0)
