# -*- coding: utf-8 -*-
from lsdecomp import admm, image_loader

#create an exapmle

loader = image_loader.ImageLoader()
frames = 47
width, height, D0 = loader.load('../datasets/ball', frames)

#implement ADMM
decomper = admm.ADMM(tau_rate = 1, beta_rate = 5, tol = 1e-6, mu_rate = 10, max_iter = 20, disp_iter = 1)
L, S = decomper.decomp_via_vector(D0, width, height, frames)

#create masks
tsd = 0.01
M = (abs(S) >= tsd) * 255

#save bg & masks
loader.save('../output/ball/background', 'bg', L * 255, width, height, frames)
loader.save('../output/ball/masks', 'mask', M, width, height, frames)