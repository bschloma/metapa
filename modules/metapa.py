import numpy as np
import os
import pickle
import datetime

class mpaParams:
    # Minimal class for keeping track of model parameters

    def __init__(self):

        self.lam = .1
        self.mu_c = .1
        self.sig2_c = .05
        self.mu_K = 100
        self.sig2_K = 70.
        self.mu_d = .1
        self.sig2_d = .05
        self.w = 1.
        self.dKcov = .02

        self.numsteps = 0
        self.numtrials = 0

        self.M = 9
        self.N = 2


"""This is the main class for the metapa model."""


class metapa:

    def __init__(self):
        """initialze metapa class."""

        # save name and location
        self.savename = str(datetime.datetime.now())
        self.savedir = str(os.getcwd())

        # mpaParams class for storing and editing parameter values
        self.params = mpaParams()

        # species field (MxMxN)
        self.s = np.array([])
        self.initS()

        # rates
        self.carr = np.array([])
        self.larr = np.array([])
        self.Karr = np.array([])
        self.darr = np.array([])

        # order parameters
        self.Marr = np.array([])
        self.alpha = np.array([])

    def initS(self):
        """intialize species field.  currently initializes randomly, uniformly.
        After changing M or N, need to call initS again to reinitialize.

        To do:  add other options, i.e. for invasion studies"""

        # fraction initialized as 1.  possibly add to mpaParams
        initThresh = .5
        compareThresh = 1 - initThresh

        # species field s, initialized with uniform random numbers
        self.s = np.random.rand(self.params.M, self.params.M, self.params.N)

        # mask with initThresh
        self.s[self.s <= compareThresh] = 0.
        self.s[self.s > compareThresh] = 1.

    def initRates(self):
        """intialize arrays with repeated values of d,K, and l, (MxMxN),
        and construct interaction matrix, c (NxN).
        To do:  test"""

        ##### compute carr from a gamma distribution.  See Dickens, Fisher, Mehta (2016) #####
        k_c = self.params.mu_c * self.params.mu_c / self.params.sig2_c / self.params.N
        theta_c = self.params.sig2_c / self.params.mu_c

        self.carr = np.random.gamma(k_c, scale=theta_c, size=(self.params.N, self.params.N))

        ### compute larr using constant lambda  ###
        self.larr = self.params.lam * np.ones((self.params.M, self.params.M, self.params.N))

        ##### compute darr and Karr from a joint lognormal distribution.  let p = (d,K) #####
        ld = np.log(
            self.params.mu_d * self.params.mu_d / np.sqrt(self.params.mu_d * self.params.mu_d + self.params.sig2_d))
        z2d = np.log(1 + self.params.sig2_d / self.params.mu_d / self.params.mu_d)

        lK = np.log(
            self.params.mu_K * self.params.mu_K / np.sqrt(self.params.mu_K * self.params.mu_K + self.params.sig2_K))
        z2K = np.log(1 + self.params.sig2_K / self.params.mu_K / self.params.mu_K)

        z2dK = np.log(1 + self.params.dKcov / self.params.mu_K / self.params.mu_d)

        cov_arr = np.array([[z2d, -z2dK],
                            [-z2dK, z2K]])

        mean_arr = np.array([ld, lK])

        # logp = np.random.multivariate_normal(mean_arr,cov_arr,size=self.params.M*self.params.M*self.params.N)
        logp = np.random.multivariate_normal(mean_arr, cov_arr, size=self.params.N)

        p = np.exp(logp)
        # self.darr = p[:,0].reshape(self.params.M,self.params.M,self.params.N)
        # self.Karr = p[:,1].reshape(self.params.M,self.params.M,self.params.N)
        Karr = p[:, 1].reshape(1, 1, self.params.N)
        self.Karr = np.tile(Karr, (self.params.M, self.params.M, 1))

        darr = p[:, 0].reshape(1, 1, self.params.N)
        self.darr = np.tile(darr, (self.params.M, self.params.M, 1))

    def update(self):
        """function for updating in one timestep"""

        ####### set things up ########

        # generate random numbers for each species at each lattice site
        random_numbers = np.random.rand(self.params.M, self.params.M, self.params.N)

        # compute eps
        ############# NEED TO INCLUDE K IN INTERACTION TERM !!!! ###########
        Keff = self.Karr - np.tensordot(self.s * self.Karr, self.carr, axes=1)
        eps = np.exp(-Keff / self.params.w)

        # compute pImmigrate
        pImmigrate = self.larr * (1 - self.s)

        # compute pExtinct
        pExtinct = eps * self.s

        # compute pDisperse
        sLeft = np.roll(self.s, 1, axis=1)
        sRight = np.roll(self.s, -1, axis=1)
        sTop = np.roll(self.s, 1, axis=0)
        sBottom = np.roll(self.s, -1, axis=0)

        pDisperse = self.darr * (1 - self.s) * (sLeft + sRight + sTop + sBottom) / 4

        ####### make transitions #######
        # print(self.s[0,0,0])
        # print(pExtinct[0,0,0])
        # immigrate
        self.s[random_numbers < pImmigrate] = 1
        # if random_numbers[0,0,0] < pImmigrate[0,0,0]:
        # print('?')
        # print(self.s[0,0,0])
        # print(pExtinct[0,0,0])

        # go extinct
        self.s[random_numbers < pExtinct] = 0
        # if random_numbers[0,0,0] < pImmigrate[0,0,0] + pExtinct[0,0,0]:
        # print('!')
        # print(self.s[0,0,0])

        # disperse
        self.s[random_numbers < pDisperse] = 1

    def evolve(self):
        """run update numsteps times."""
        for n in range(self.params.numsteps):
            self.update()

    def simulate(self):
        """evolve for numsteps and numtrials.  compute order parameters"""
        Mmat = np.zeros((self.params.numsteps, self.params.numtrials))
        running_time_average = np.zeros((self.params.M, self.params.M, self.params.N, self.params.numtrials))
        for trial in range(self.params.numtrials):
            self.initS()
            self.initRates()

            for step in range(self.params.numsteps):
                Mmat[step, trial] = np.sum(np.sum(np.sum(self.s))) / self.params.M / self.params.M
                # alpha1_mat[step,trial] = 4/self.params.N*np.sum(np.sum(np.sum(self.s*self.s)))
                # alpha2_mat[step,trial] = 4/self.params.N*np.sum(np.sum(np.sum(self.s)))

                running_time_average[:, :, :, trial] = self.s
                if step > 200:
                    running_time_average[:, :, :, trial] = (running_time_average[:, :, :, trial] + self.s) / 2

                # species1mat[step,trial] = self.s[0,0,0]
                self.update()

        self.Marr = np.mean(Mmat, axis=1)

        #
        self.alpha = np.mean(running_time_average * running_time_average, axis=3) - np.mean(running_time_average,
                                                                                            axis=3) * np.mean(
            running_time_average, axis=3)

        # self.species1mat = species1mat

        def save(self):
            # save function using the pickle package

            savestr = self.savedir + os.sep + self.savename
            with open(savestr, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


'''function for computing composite order parameter from M and alpha.

   I cant figure out linear indexing right now cuz theres no wifi, 
   but that would be a better way of doing this.'''


def computeC(mpa, Marr, alpha, gamma):
    C = np.zeros(Marr.shape)
    g = np.zeros(Marr.shape)

    if np.array(Marr.shape).size == 2:
        for m in range(Marr.shape[0]):
            for n in range(Marr.shape[1]):
                if Marr[m, n] > gamma:
                    g[m, n] = (gamma - Marr[m, n]) / (mpa.params.N - gamma)
                else:
                    g[m, n] = (gamma - Marr[m, n]) / gamma
                C[m, n] = (1 - alpha[m, n] * 4 / mpa.params.N / mpa.params.M / mpa.params.M) * g[m, n]
    elif np.array(Marr.shape).size == 1:
        for m in range(Marr.shape[0]):
            if Marr[m] > gamma:
                g[m] = (gamma - Marr[m]) / (mpa.params.N - gamma)
            else:
                g[m] = (gamma - Marr[m]) / gamma
            C[m] = (1 - alpha[m] * 4 / mpa.params.N / mpa.params.M / mpa.params.M) * g[m]

    return C




