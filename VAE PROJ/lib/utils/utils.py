import matplotlib.pyplot as plt
import torch


def show(samples, name='Samples', fix=False, ax=None):
    Xs, Ys, Zs= samples[:,0], samples[:,1], samples[:,2]

    if ax is not None:
        painter = ax
        painter.scatter(Xs, Ys, Zs)
        painter.set_title(name)
    else:
        painter = plt
        painter.figure(123, figsize=(8,8))

        painter.clf()
        painter.scatter(Xs, Ys,Zs)
        painter.xlim(-15, 15)
        painter.ylim(-15, 15)
        painter.title(name)

        if not fix:
            painter.pause(.002)
        else:
            painter.show()


def get_device_str():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'