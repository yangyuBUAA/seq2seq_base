from torch.nn import CrossEntropyLoss
from torch.nn.functional import nll_loss
import torch

if __name__ == '__main__':
    lossf = CrossEntropyLoss()
    a = torch.tensor(
       [[[0.1, 0.2, 0.3, 0.1, 0.3],
         [0.5, 0.1, 0.1, 0.2, 0.1],
         [0.8, 0.1, 0.04, 0.03, 0.03]],

        [[0.4, 0.1, 0.1, 0.2, 0.2],
         [0.5, 0.4, 0.01, 0.08, 0.01],
         [0.4, 0.2, 0.1, 0.1, 0.2]],

        [[0.4, 0.4, 0.1, 0.05, 0.05],
         [0.1, 0.7, 0.1, 0.05, 0.05],
         [0.3, 0.2, 0.1, 0.2, 0.2]]
       ]
    )
    a = a.permute(0, 2, 1)
    print(a.shape)
    b = torch.tensor(
        [[1, 4, 2],
         [2, 3, 0],
         [1, 2, 1]]
    )
    c = torch.log(a)
    print(c)
    print(b.shape)
    print(nll_loss(c, b))
    # shape of c: (batch_size, vocab_size, trg_len)
    # shape of b: (batch_size, trg_len)
    import numpy as np
    print((2*np.log(0.2) + 3*np.log(0.1) + 2*np.log(0.4) + np.log(0.04) + np.log(0.08))/9)