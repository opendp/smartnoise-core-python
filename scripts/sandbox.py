import torch

# # test grad hook
# v = torch.tensor([0., 0., 0.], requires_grad=True)
# y = torch.sum(v) + torch.sum(v)
# h = v.register_hook(lambda grad: print('x'))  # double the gradient
# y.backward(torch.tensor(1.))
# print(v.grad)

# paths = 4
# Bs = [torch.rand([3, 2, 1]) for _ in range(1)]
# As = [torch.rand([3, 2, 3]) for _ in range(1)]
#
# A = As[0]
# B = Bs[0]
#
# x1 = torch.einsum('n...i,n...j->n...ij', B, A)
# x1 = torch.einsum('n...ij->nij', x1)
#
# A2 = torch.einsum('n...i->ni', A)
# B2 = torch.einsum('n...i->ni', B)
# x2 = torch.einsum('ni,nj->nij', B2, A2)
#
# print(torch.allclose(x1, x2))
# print(x1, x2)
import math

import matplotlib.pyplot as plt

data = [12799.484, 13692.874, 10817.933, 9763.151, 10741.716, 10290.911, 10725.086, 10787.445, 10880.651, 11086.131, 10994.266, 10762.792, 10907.375, 11149.268, 11251.668, 11341.082, 10816.922, 10995.322, 10928.616, 11086.191, 10671.836, 10707.416, 10703.940, 10798.393, 10916.990, 10977.482, 11109.494, 11242.994, 11264.267, 10981.552, 10889.118, 11017.144, 11013.825, 10785.565, 10837.913, 10893.697, 10993.397, 11084.676, 11153.987, 10946.433, 10740.991, 10845.013, 10655.940, 10686.067, 10752.116, 10728.720, 10683.428, 10623.577, 10685.491, 10738.990, 10747.232]

plt.plot(data)
plt.show()



# for A, B in zip(As, Bs)