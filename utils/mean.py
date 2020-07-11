def get_mean(norm_value=255, dataset='std'):
    assert dataset in ['quva', 'std']

    if dataset == 'quva':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'std':
        return [127.0 / norm_value, 127.0 / norm_value, 127.0 / norm_value]
        # return [0.0 / norm_value, 0.0 / norm_value, 0.0 / norm_value]

def get_std(norm_value=255, dataset = 'std'):
    assert dataset in ['quva', 'std']

    if dataset == 'quva':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'std':
        return [127.0 / norm_value, 127.0 / norm_value, 127.0 / norm_value]
        # return [1.0 / norm_value, 1.0 / norm_value, 1.0 / norm_value]

