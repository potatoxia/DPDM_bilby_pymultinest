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




arg_nsource=sys.argv[1]#sys.argv[1] #'6'
arg_model=sys.argv[2]#sys.argv[2] #'H1'
fmax=float(sys.argv[3])
fmin=1e-9


def fre(log_ma):
    f=10**log_ma/(2*np.pi)*1.519e15
    return f


def log_m(f):
    m=np.log10(f/1.519e15*2*np.pi)
    return m


log_ma_max=log_m(fmax)
log_ma_min=log_m(fmin)
print("f")
print(fmin,fmax)
print("log_ma")
print(log_ma_min,log_ma_max)


datadir = '/home/xia/PTA/fit1data/'
outd='/home/xia/PTA/resultmn/'#test1/'
parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))



# pulsars used in 11 year analysis
# filter
p9 = np.loadtxt(datadir+'PPTA_pulsars'+arg_nsource+'.txt', dtype=str)
nsource=np.size(p9)
parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in p9]
timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in p9]

####1 pulsar
#psrs = Pulsar(parfiles[0],timfiles[0])


####more pulsars
psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)






###setup model
dp = DPDM.dpdm_block(type_ = 'Bayes')
tm = gp_signals.TimingModel()
wnb = models.white_noise_block(vary=False)
dmn = models.dm_noise_block(components=30)
spn = models.red_noise_block(components=30)
if arg_model=='H1':model = tm + dp + wnb + dmn + spn

if arg_model=='H0': model = tm + wnb + dmn + spn


#########
#
#if nsource==1:
#    signals = model(psrs)
#    pta = signal_base.PTA(signals)
#    xs = {par.name: par.sample() for par in pta.params}
#    lnlike=pta.get_lnlikelihood(xs)
#    nparams=len(signals.params)
#    print(lnlike,p9)





signals = []
for i in range(nsource):
    signal = model(psrs[i])
    signals.append(signal)
    pta = signal_base.PTA(signal)
    print(p9[i])



    
    
pta = signal_base.PTA(signals)


ndim = len(pta.params)
noisepars = np.load(datadir+'M30/noisepars.npy')
noiseparsname = np.load(datadir+'M30/noisepardict.npy')
noise_dict = {noiseparsname[i]:noisepars[i] for i in range(len(noisepars))}



#def test(name):
#    xs = {par.name: par.sample() for par in pta.params}
#    x0 = {**noise_dict,**xs}
#    try: return xs[name],noise_dict[name],x0[name]
#    except KeyError: return noise_dict[name],x0[name]


#test('J2241-5236_red_noise_log10_A')
#test('J0437-4715_10CM_CPSR2_efac')



def mylnlikelihood(xss):
    x0 = {**noise_dict,**xss}
    return pta.get_lnlikelihood(x0)



xs = {par.name: par.sample() for par in pta.params}
mylnlikelihood(xs)




ndims=len(pta.params)
priors = {}
names=[0 for i in range(ndims)]
ndims=len(pta.params)
pmins=np.ones(ndims)
pmaxs=np.ones(ndims)
for i in range(ndims):
    namei=pta.params[i].name
    names[i]=namei
    if namei=='x_dp_log10_ma':pmins[i],pmaxs[i]=log_ma_min,log_ma_max
    else:pmins[i],pmaxs[i]=pta.params[i]._pmin, pta.params[i]._pmax
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
if arg_model=='H1': label = "name"+arg_nsource+'_'+arg_model+'_'+str(fmin)+'_'+str(fmax)

if arg_model=='H0': label = "name"+'_'+arg_nsource+'_'+arg_model
#setup_logger(outdir=outdir, label=label)




result = run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=400, outdir=outd, label=label, use_ratio=False, resume = False, verbose = True)
#result.plot_corner()





