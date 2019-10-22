from __future__ import division

import numpy as np
np.seterr(divide = 'ignore')
import glob
#import matplotlib.pyplot as plt
import scipy.linalg as sl
import sys
#sys.path.append('~/anaconda3/lib/python3.7/site-packages/libstempo')
sys.path.append('~/anaconda3/lib/python3.7/site-packages/enterprise-1.2.0-py3.7.egg/enterprise')
#sys.path.append('/Users/xiahe/Desktop/PTA/DPDM')
#from libstempo.libstempo import *
#import libstempo
#libstempo.__path__
#import libstempo as T

try:
    from mpi4py import MPI
except ImportError:
    print('Do not have mpi4py package.')
    from . import nompi4py as MPI


print(MPI.COMM_WORLD.Get_rank())
print(MPI.COMM_WORLD.Get_size())

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise_extensions as ee
from enterprise_extensions import models
from enterprise_extensions import DPDM

import corner
#from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import inspect
from bilby.core.prior import Uniform
from bilby.core.utils import setup_logger
from bilby import Likelihood, run_sampler
import pymultinest




###################################################
##########Get par, tim files############
#########Load into Pulsar class list###############
#datadir = '/Users/xiahe/Desktop/PTA/fit1data/'
#outd='/Users/xiahe/Desktop/PTA/resultmn/'



datadir = '/home/xia/PTA/fit1data/'
outd='/home/xia/PTA/resultmn/'#test1/'
parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))



# 18 pulsars used in 9 year analysis
# filter
p9 = np.loadtxt(datadir+'PPTA_pulsars1.txt', dtype=str)
parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in p9]
timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in p9]

psrs = Pulsar(parfiles[0],timfiles[0])

#psrs = []
#for p, t, pu in zip(parfiles,timfiles,p9):
#    psr0 = Pulsar(datadir+str(pu)+'.par', datadir+str(pu)+'.tim')#, ephem='DE414')
#    psrs.append(psr0)


###setup model
dp = DPDM.dpdm_block(type_ = 'Bayes')
tm = gp_signals.TimingModel()
#wnb = models.white_noise_block(vary=False)
wnb = models.white_noise_block(vary=True)
dmn = models.dm_noise_block(components=30)
spn = models.red_noise_block(components=30)
#model = tm + dp + wnb + dmn + spn
model = tm + wnb + dmn + spn

#########
if np.size(psrs)==1:
    signals = model(psrs)
    pta = signal_base.PTA(signals)
    xs = {par.name: par.sample() for par in pta.params}
    lnlike=pta.get_lnlikelihood(xs)
    nparams=len(signals.params)
    print(lnlike,p9)

#
#lnlike=np.ones(len(psrs))
#for i in range(len(psrs)):
#    signal = model(psrs[i])
#    pta = signal_base.PTA(signal)
#    xs = {par.name: par.sample() for par in pta.params}
#    lnlike[i]=pta.get_lnlikelihood(xs)
#    print(lnlike[i],p9[i])
#
#
#nparams = [] # Get the number of parameters of each pulsar
#signals = []
#for psr in psrs:
#    signal = model(psr)
#    nparams.append(len(signal.params)) # Subtracting common DPDM params
#    signals.append(signal)
#
#pta = signal_base.PTA(signals)


if np.size(psrs)==1:
    signals = model(psrs)
    pta = signal_base.PTA(signals)
    xs = {par.name: par.sample() for par in pta.params}
    lnlike=pta.get_lnlikelihood(xs)
    nparams=len(signals.params)
    print(lnlike,p9)




ndim = len(pta.params)
xs = {par.name: par.sample() for par in pta.params}
x0 = np.hstack(p.sample() for p in pta.params)
pta.get_lnlikelihood(xs)
pta.get_lnlikelihood(x0)
len(pta.params)
len(xs)



ndims=len(pta.params)
priors = {}
names=[0 for i in range(ndims)]
ndims=len(pta.params)
pmins=np.ones(ndims)
pmaxs=np.ones(ndims)
for i in range(ndims):
    namei=pta.params[i].name
    names[i]=namei
    pmins[i]=pta.params[i]._pmin
    pmaxs[i]=pta.params[i]._pmax
    priors[namei]=Uniform(pmins[i], pmaxs[i], name=namei)






class own_likelihood(Likelihood):
    def __init__(self,func):
        self.callfunc = func
        parameters = names
        self.parameters = dict.fromkeys(parameters)
    def log_likelihood(self):
        return self.callfunc(self.parameters)


likelihood = own_likelihood(pta.get_lnlikelihood)




#outdir = "outdir"
label = "name0"
#setup_logger(outdir=outdir, label=label)




result = run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=100, outdir=outd, label=label, use_ratio=False, resume = False, verbose = True)
#result.plot_corner()





