import sys 
sys.path.append('../enterprise/enterprise') 
from pulsar import Pulsar
import glob
from enterprise.signals import signal_base
from enterprise.signals import utils
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise_extensions import models
from enterprise_extensions import DPDM
from enterprise.signals import selections


import numpy as np
np.seterr(divide = 'ignore')

from PTMCMCSampler.PTMCMCSampler import PTSampler

import multiprocessing


"""
Searching for DPDM signal using the Frequentist method in any PTA data release.

Xiao Xue (2019.09)

xuexiao@mail.itp.ac.cn

"""






if __name__ == '__main__':


	# Importing files.

	datadir = '../ppta/fit1data'
	parfiles = sorted(glob.glob(datadir + '/*.par'))
	timfiles = sorted(glob.glob(datadir + '/*.tim'))
	parfiles.remove(datadir+'/J1125-6014.par')
	timfiles.remove(datadir+'/J1125-6014.tim')

		# I throw away this certain pulsar because it may return a unstable, sometimes
		# very large ln-likelihood value. I think there is a bug somewhere.
	

	psrs = []
	for p, t in zip(parfiles, timfiles):
		psr = Pulsar(p, t)
		psrs.append(psr)


	save1 = np.load('M30/noisepars.npy')
	save2 = np.load('M30/noisepardict.npy')
	Dict = {save2[i]:save1[i] for i in range(len(save1))}

	# The Big Model
	# dm noise
	log10_A_dm = parameter.Constant()
	gamma_dm = parameter.Constant()
	pl_dm = utils.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
	dm_basis = utils.createfourierdesignmatrix_dm(nmodes=30,
							Tspan=None)
	dmn = gp_signals.BasisGP(pl_dm, dm_basis, name='dm_gp',
					coefficients=False)
	# spin noise
	log10_A = parameter.Constant()
	gamma = parameter.Constant()
	pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
	selection = selections.Selection(selections.no_selection)
	spn = gp_signals.FourierBasisGP(pl, components=30, Tspan=None,
					coefficients=False, selection=selection,
					modes=None)
		


	dp = DPDM.dpdm_block(type_ = 'Bayes')
	tm = gp_signals.TimingModel()
	wnb = models.white_noise_block(vary=False)


	model = tm + dp + wnb + dmn + spn
	nparams = [] # Get the number of parameters of each pulsar
	signals = []
	for psr in psrs:
		signal = model(psr)
		nparams.append(len(signal.params)-5) # Subtracting common DPDM params
		signals.append(signal)
	PTA = signal_base.PTA(signals)
	ndim = len(PTA.params)

	# Use the best fit noise parameters!
	PTA.set_default_params(Dict)

	x0 = np.hstack([par.sample() for par in PTA.params])


	sampler = PTSampler(ndim,PTA.get_lnlikelihood,PTA.get_lnprior,
				cov = np.diag(np.ones(ndim)*0.25), groups=None, outDir='/home/sdb/xuexiao/M30PTAchains/FixNoiseBayesian/')
	sampler.sample(x0,10000000,isave=1000)






