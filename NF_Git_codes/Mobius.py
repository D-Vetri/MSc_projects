import torch
from torch import nn
import conditional as cd

def mobius(config,feature_dim):
        return MobiusFlow(config.D,
                          config.segments,
                          condition=config.condition,
                          feature_dim=feature_dim)


def _h(z,w,D=3):
    w_norm = torch.norm(w, dim=-1, keepdim=True)  # n x k x 1

    h_z = (1 - w_norm**2) / (
        torch.norm((z.reshape(-1, 1, D) - w), dim=-1, keepdim=True) ** 2
    ) * (z.reshape(-1, 1, D) - w) - w
    return h_z

class MobiusFlow(nn.Module):
    def __init__(self,D,k,condition=False,feature_dim=None):
        super().__init__()
        
        self.D = D
        self.k = k

        self.condition = condition
        self.feature_dim = feature_dim

        if self.condition:
            input_dim = D + feature_dim
        else:
            input_dim = D
        
        self.conditioner = cd.ConditionalTransform(input_dim,4*k) #1K for the alphas and 3K, one for each element in the transformation orthonormal vector(c2)
        self.register_buffer("_I",torch.eye(D), persistent=False)

    def forward(self,rotations, permute=None,feature=None):
        assert permute != None,"Please provide permuting for proper expressivity"
        if self.condition:
            assert feature != None,"You have not provided features for conditional transformation"             
        x = rotations[...,permute[0]] #[k,3] so3 columns represented as rows
        c1 = rotations[...,permute[1]] # [k,3]
        
        if self.condition:
            condition_input = torch.cat((c1,feature),dim=-1)
        else:
            condition_input = c1
        
        conditions = self.conditioner(condition_input)
        
        alpha,w = torch.split(conditions,[self.k, 3* self.k],dim=1)
        w = w.reshape(-1,self.k,self.D)
        
        proj = self._I[None,...] - torch.einsum('ni,nj->nij',c1,c1) # Gram-Schmidt projection. 
        w = torch.einsum('nij,nkj->nki',proj,w)
        c2 = -x # this is needed to ensure that the manifoldal smoothness remains intact(the transformation ensure SO3 rules remain)
        c2 = c2/c2.norm(dim=-1,keepdim=True)
        c3 = torch.linalg.cross(c1,c2)
        c3 = c3/c3.norm(dim=-1,keepdim=True)

        alpha = torch.nn.functional.softplus(alpha)
        sum_alpha= alpha.sum(dim=-1,keepdim=True)
        alpha = alpha/sum_alpha
        w = 0.7/(1+torch.norm(w,dim=-1,keepdim=True))*w #To prevent angle discontinuities
        
        tc2,ldj = self._transform(x,c2,c3,alpha,w)
        if (permute[1]-permute[0])==1 or (permute[1]-permute[0])==-2:
            tc3 = torch.linalg.cross(tc2,c1)
        else:
            tc3 = torch.linalg.cross(c1,tc2)
        tc3 =tc3/tc3.norm(dim=-1,keepdim=True)
        trotations = torch.empty(rotations.size()).to(rotations.device)
        trotations[...,permute[0]] = tc2
        trotations[...,permute[1]] = c1
        trotations[...,permute[2]] = tc3

        return trotations,ldj
    


    def _transform(self,x,c2,c3,alpha,w):
        z = x

        h_z = _h(z,w,self.D)
        radians = torch.atan2(
            torch.einsum('nki,ni->nk',h_z,c3),
            torch.einsum('nki,ni->nk',h_z,c2)
        )
        tc2 = radians #angle for the new position of c2, that is c2' 
        tc2 = torch.where(tc2>=0,tc2,tc2+(2*torch.pi))
        tc2 = torch.sum(alpha*tc2,dim=1,keepdim=True)
        
        tc2 = c2*torch.cos(tc2) + c3*torch.sin(tc2) #new c2
        
        #Jacobian calculations
        z_w = z[:,None,:] - w
        z_w_norm = torch.norm(z_w,dim=-1)
        z_w_unit = z_w/z_w_norm[...,None]

        theta = torch.atan2(
            torch.einsum('ni,ni->n',x,c3),
            torch.einsum('ni,ni->n',x,c2)
        ).reshape(-1,1) #single dimensional theta(scalar but with a dim)

        dz_dtheta = -torch.sin(theta)*c2 + torch.cos(theta) * c3

        dh_dz = (
            (1-torch.norm(w,dim=-1)**2)[...,None,None] # jacobian in the paper
            *(self._I[None,None,...]-2*torch.einsum('nki,nkj->nkij',z_w_unit,z_w_unit))
            /(z_w_norm[...,None,None]**2)        
        )
        
        #batch wise matrix-vector multiplication 
        dh_dtheta = torch.einsum("nkpq,nq->nkp",dh_dz,dz_dtheta)

        # dc'/dtheta that is |det J|
        dtc2 = torch.sum(torch.norm(dh_dtheta,dim=-1)*alpha,dim=1)
        return tc2,torch.log(dtc2)
    
    def inverse(self, trotation, permute=None, feature=None):
        assert permute != None, "Please provide permuting for proper expressivity"
        if self.condition:
            assert feature != None, "You have not provided features for \
                conditional transformation"

        tx = trotation[..., permute[0]]
        tc1 = trotation[..., permute[1]]

        if self.condition:
            condition_input=torch.cat((tc1, feature),dim=-1)
        else:
            condition_input=tc1

        conditions = self.conditioner(condition_input)
        alpha, w = torch.split(
            conditions, [self.k, 3 * self.k], dim=1
        )
        w = w.reshape(-1, self.k, self.D)
        proj = self._I[None, ...] - torch.einsum("ni,nj->nij", tc1, tc1)
        w = torch.einsum("nij,nkj->nki", proj, w)

        tc2 = -tx
        tc2 = tc2/tc2.norm(dim=-1, keepdim=True)
        tc3 = torch.linalg.cross(tc1, tc2)
        tc3 = tc3/tc3.norm(dim=-1, keepdim=True)
        alpha = torch.nn.functional.softplus(alpha)
        sum_alpha = alpha.sum(dim=-1, keepdim=True)
        alpha = alpha / sum_alpha
        w = 0.7  / (1 + torch.norm(w, dim=-1, keepdim=True)) * w

        ttheta = torch.atan2(
            torch.einsum("ni,ni->n", tx, tc3), torch.einsum("ni,ni->n", tx, tc2)
        ).reshape(-1, 1)
        ttheta = torch.where(ttheta >= 0, ttheta, (ttheta + 2*torch.pi))

        ttheta = torch.where(
            abs(ttheta - torch.pi * 2) < 1e-4,
            torch.zeros(ttheta.size(), dtype=ttheta.dtype,
                        device=ttheta.device),
            ttheta,
        )

        theta = self._bin_find_root(ttheta, tc2, tc3, alpha, w)
        c2 = tc2* torch.cos(theta) + tc3 * torch.sin(theta)
        _, ldj = self._transform(c2, tc2, tc3, alpha, w)

        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            c3 = torch.linalg.cross(c2, tc1)
        else:
            c3 = torch.linalg.cross(tc1, c2)
        c3 = c3/c3.norm(dim=-1, keepdim=True)
        rotation = torch.empty(trotation.size()).to(trotation.device)
        rotation[..., permute[0]] = c2
        rotation[..., permute[1]] = tc1
        rotation[..., permute[2]] = c3
        
        return rotation, -ldj

    def _bin_find_root(self, y, r, v, alpha, w):
        return BinFind.apply(y, r, v, alpha, w)


class BinFind(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, r, v, alpha, w):
        ctx.r = r.clone().detach()
        ctx.v = v.clone().detach()
        ctx.alpha = alpha.clone().detach()
        ctx.w = w.clone().detach()
        a = (
            torch.ones(y.size(), device=y.device, dtype=y.dtype)
            * torch.pi
            / 2
        )
        b = (
            torch.ones(y.size(), device=y.device, dtype=y.dtype)
            * 3
            / 2
            * torch.pi
        )
        time = 1
        while abs(torch.max(b - a)) >= 1e-4:
            x0 = (a + b) / 2
            fx0 = BinFind._forward_theta(x0, r, v, alpha, w) - y

            if time > 100:
                print("fail")
                break

            bigger = fx0 < 0
            lesser = fx0 >= 0
            a = a + (b - a) / 2 * bigger
            b = b - (b - a) / 2 * lesser

            time += 1
        ctx.x = x0.clone().detach()
        ctx.y = y.clone().detach()
        return x0

    @staticmethod
    def _h(z, w, D=3):
        return _h(z, w, D)

    @staticmethod
    def _forward_theta(x, r, v, alpha, w):
        """input: theta, return theta' and partial theta'/ partial theta
        used to compute inverse"""
        z = r * torch.cos(x) + v * torch.sin(x)

        h_z = BinFind._h(z, w)
        radians = torch.atan2(
            torch.einsum("nki,ni->nk", h_z,
                         v), torch.einsum("nki,ni->nk", h_z, r)
        )
        tx = radians 
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)
        tx = torch.sum(alpha * tx, dim=1, keepdim=True)

        return tx

    @staticmethod
    def backward(ctx, x_grad):
        x = ctx.x
        y = ctx.y
        r = ctx.r
        v = ctx.v
        alpha = ctx.alpha
        w = ctx.w
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            r.requires_grad_(True)
            v.requires_grad_(True)
            alpha.requires_grad_(True)
            w.requires_grad_(True)
            x_grad_2, r_grad, v_grad, alpha_grad, w_grad = torch.autograd.grad(BinFind._forward_theta(
                x, r, v, alpha, w), (x, r, v, alpha, w), torch.ones_like(x_grad))
            y_grad = torch.where(x_grad_2 != 0, 1/x_grad_2,
                                 torch.zeros_like(x_grad_2))*x_grad
            r_grad = torch.where(x_grad_2 != 0, -r_grad /
                                 x_grad_2, torch.zeros_like(r_grad))*x_grad
            v_grad = torch.where(x_grad_2 != 0, -v_grad /
                                 x_grad_2, torch.zeros_like(v_grad))*x_grad
            w_grad = torch.where(x_grad_2.unsqueeze(-1) != 0, -w_grad /
                                 x_grad_2.unsqueeze(-1), torch.zeros_like(w_grad))*x_grad.unsqueeze(-1)
            alpha_grad = torch.where(
                x_grad_2 != 0, -alpha_grad/x_grad_2, torch.zeros_like(alpha_grad))*x_grad
        return y_grad, r_grad, v_grad, alpha_grad, w_grad
