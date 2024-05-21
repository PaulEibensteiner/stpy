import cvxpy as cp
import numpy as np
import torch
from stpy.borel_set import BorelSet
import matplotlib.pyplot as plt
from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, ConcatEmbedding, MaskedEmbedding
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval, interval_torch
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.continuous_processes.regularized_dictionary_psd import RegularizedDictionaryPSD
from stpy.regularization.sdp_constraint import SDPConstraint, SDPLinearInequality

if __name__ == "__main__":


    N = 12
    n = 128
    d = 1
    eps = 0.01
    s = 0.01
    m = 64
    for sign in [1,-1]:
        f = lambda x: (0.5 * torch.sin(x * 20)**2 * (sign* x > 0).double()+\
                      0.25 * torch.sin(x * 10)**2 * (sign*x > 0).double()+\
                      0.25 * torch.sin(x * 5)**2 * (sign*x > 0).double() )

        #Xtrain = interval_torch(n=N, d=1)
        Xtrain = interval_torch(n=N, d=1, L_infinity_ball=0.4)+0.6*sign
        ytrain = f(Xtrain)

        torch.manual_seed(22)
        np.random.seed(22)

        xtest = torch.from_numpy(interval(n, d, L_infinity_ball=1))
        kernel_object = KernelFunction(gamma=0.03, d=1)
        xtest_nystrom = torch.from_numpy(interval(m, d, L_infinity_ball=1))

        embedding = NystromFeatures(kernel_object=kernel_object, m = m)
        embedding.fit_gp(xtest_nystrom, None)


        likelihood = GaussianLikelihood(sigma=s)
        cmap = plt.get_cmap('viridis')
        num_colors = 1
        max_m = 32
        min_m = 2
        colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
        pairs = list(zip(np.arange(min_m,max_m+1,max_m//num_colors), colors))

        S1 = BorelSet(1, torch.Tensor([[-1, -0.]]))
        S2 = BorelSet(1, torch.Tensor([[ 0., 1.]]))
        S3 = BorelSet(1, torch.Tensor([[ -1., 1.]]))

        C = 2
        phi1 = C*embedding.integral(S1)
        phi2 = C*embedding.integral(S2)
        phi3 = embedding.integral(S3)

        L = (phi1 - phi2).T @ (phi1 - phi2) + (phi2 - phi1).T @ (phi2 - phi1)
        L2 = phi3.T @ phi3

        linear_constraint = SDPLinearInequality(Bs = [-L],cs = [1000.7] )

        regularization = SDPConstraint(trace_constraint = 505.)

        regularization2 = SDPConstraint(trace_constraint=505.,
                                        linear_constraint=linear_constraint, regularization=False)

        estimator = RegularizedDictionaryPSD(embedding, likelihood,
                                             sdp_constraint=regularization2,
                                             norm_regularization=False)

        estimator_classical = RegularizedDictionaryPSD(embedding, likelihood,
                                                       sdp_constraint=regularization,
                                                       norm_regularization=False)


        estimator.load_data((Xtrain, ytrain))
        estimator.fit()
        estimator_classical.load_data((Xtrain, ytrain))
        estimator_classical.fit()


        mu = estimator.mean(xtest)
        mu_classical = estimator_classical.mean(xtest)

        print("ESTIMATOR DIAGONAL")

        print("diag:",torch.diag(estimator.A_fit))
        print("trace:",torch.trace(estimator.A_fit))
        print("norm:",torch.linalg.norm(estimator.theta_fit))
        print ("constraint:",torch.trace(L @ estimator.A_fit))

        print("ESTIMATOR CLASSICAL")

        print("diag:",torch.diag(estimator_classical.A_fit))
        print("teace:",torch.trace(estimator_classical.A_fit))
        print("norm:",torch.linalg.norm(estimator_classical.theta_fit))

        plt.plot(xtest, mu,  lw=3, label='new', color='blue')
        plt.plot(xtest, mu_classical, lw=3, label='classical', color='yellow', linestyle = '-')

        #lcb = estimator.lcb(xtest)
        #ucb = estimator.ucb(xtest)
        #bias = estimator.bias(xtest)
        #plt.fill_between(xtest.view(-1), lcb.view(-1), ucb.view(-1), color='blue', alpha=.1)
        #plt.fill_between(xtest.view(-1), mu.view(-1) - bias.view(-1), mu.view(-1) + bias.view(-1), color='blue', alpha=.1)

        plt.plot(Xtrain, ytrain, 'ko', lw=3)
        plt.plot(xtest, f(xtest), 'k--', lw=3)

        plt.legend()
        plt.savefig("new_psd_min_inequality_{}.pdf".format(sign), bbox_inches="tight")

        #plt.show()
        plt.clf()