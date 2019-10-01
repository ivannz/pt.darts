""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from scipy.sparse.linalg import cg, LinearOperator
import numpy as np

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay, hessian_vector_type = 1):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.hessian_vector_type = hessian_vector_type

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # compute gradient

        def ddot(a, b):
           return sum(u.flatten() @ v.flatten() for u, v in zip(a, b))

        if self.hessian_vector_type in [0, 1, 2]:
            # do virtual step (calc w`)
            self.virtual_step(trn_X, trn_y, xi, w_optim)

            # calc unrolled loss
            val_loss = self.v_net.loss(val_X, val_y) # L_val(w`)

            v_alphas = tuple(self.v_net.alphas())
            v_weights = tuple(self.v_net.weights())
            v_grads = torch.autograd.grad(val_loss, v_alphas + v_weights)
            dalpha = v_grads[:len(v_alphas)] # d_a L_val(w')
            dw = v_grads[len(v_alphas):]     # d_w L_val(w')

        #
        # DARTS
        #
        if self.hessian_vector_type == 1:
            hessian_vector = self.compute_hessian_vector(dw, trn_X, trn_y)

        #
        # Exact hessian*grad
        #
        elif self.hessian_vector_type == 2:
            tr_loss = self.net.loss(trn_X, trn_y) # L_train(w)
            grad_L_train_w = torch.autograd.grad(tr_loss, self.net.weights(), create_graph = True)
            grad_vector = ddot(grad_L_train_w, dw)

            hessian_vector = torch.autograd.grad(grad_vector, self.net.alphas(), retain_graph = False)

        #
        # HOAG (hessian * inverse_hessian * grad)
        #
        elif self.hessian_vector_type == 3:
            # calc unrolled loss
            val_loss = self.net.loss(val_X, val_y) # L_val(w)

            # compute gradient
            v_alphas = tuple(self.net.alphas())
            v_weights = tuple(self.net.weights())
            v_grads = torch.autograd.grad(val_loss, v_alphas + v_weights)
            dalpha = v_grads[:len(v_alphas)] # d_a L_val
            dw = v_grads[len(v_alphas):]     # d_w L_val

            tr_loss = self.net.loss(trn_X, trn_y) # L_train(w)
            grad_L_train_w = torch.autograd.grad(tr_loss, self.net.weights(), create_graph = True)

            device = tr_loss.device

            N = sum([x.numel() for x in dw])

            def to_flat(t):
                return torch.cat([x.flatten().cpu() for x in t], 0)

            def calc_huge_hessian_vector(z):

                grad_vector = 0
                idx = 0
                z_pt = torch.Tensor(z).to(device)
                items = []

                tr_loss = self.net.loss(trn_X, trn_y) # L_train(w)
                grad_L_train_w = torch.autograd.grad(tr_loss, self.net.weights(), create_graph = True) # d_w L_train(w)

                for elem in grad_L_train_w:
                    grad_vector += elem.flatten() @ z_pt[idx : idx + elem.numel()]
                    idx += elem.numel()

                huge_hessian_vector = torch.autograd.grad(grad_vector, self.net.weights(), retain_graph = False)

                return to_flat(huge_hessian_vector)

            LinOp = LinearOperator((N, N), matvec = calc_huge_hessian_vector, dtype = np.dtype('float64'))

            inv_hessian_vector = cg(LinOp, to_flat(dw), tol = 1e-3, maxiter = 5)[0]
            inv_hessian_vector = torch.Tensor(inv_hessian_vector).to(device)

            #
            #  Restore tensor dimensions
            #
            inv_hessian_vector_torch = []
            idx = 0

            for elem in dw:
                n = elem.numel()
                shape = elem.shape
                inv_hessian_vector_torch.append(inv_hessian_vector[idx : idx + n].reshape(shape))
                idx += n

            hessian_vector = self.compute_hessian_vector(inv_hessian_vector_torch, trn_X, trn_y)
        else:
            hessian_vector = 0

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            if self.hessian_vector_type == 0:
                for alpha, da in zip(self.net.alphas(), dalpha):
                    alpha.grad = da
            elif self.hessian_vector_type in [1, 2]:
                for alpha, da, h in zip(self.net.alphas(), dalpha, hessian_vector):
                    alpha.grad = da - xi*h
            elif self.hessian_vector_type == 3:
                for alpha, da, h in zip(self.net.alphas(), dalpha, hessian_vector):
                    alpha.grad = da - h


    def compute_hessian_vector(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
