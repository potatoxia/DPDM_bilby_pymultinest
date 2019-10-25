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





def get_pulsar_noise(pta , ret): 

	ndim = len(pta.params)

	# Generate jump groups. 
		
	groups0 = [[i,i+1] for i in range(0,ndim-1,2)]	
	groups0.extend([range(ndim)])

	groups1 = [range(ndim)]

	# This definiton can ensure a fast convergence (I tried).



	# Start the first run. Notice the choice of starting points.

	x0 = np.zeros(ndim)
	x0[0:-4:2]=1.
	x0[1:-4:2]=-7.
	x0[-4:]=[2,-13.5,2,-13.5]
	cov0 = np.diag(np.ones(ndim)*0.05)
	outDir0='/home/sdb/xuexiao/M30PTAchains/NoiseFixing/FirstRun/'+pta.pulsars[0]
	sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior, 
		cov0, groups=groups0 , outDir = outDir0, verbose=False)
	sampler.sample(x0, 250000,isave=1000)
	chain0 = np.loadtxt(outDir0+'/chain_1.txt')

	# End of the first run.

	# Start the second run.

	outDir1='/home/sdb/xuexiao/M30PTAchains/NoiseFixing/SecondRun/'+pta.pulsars[0]
	x1 = chain0[np.where(chain0==np.max(chain0[:,-3]))[0][0],:-4]	# Get the best fit parameters from the last run.
	cov1 = np.load(outDir0 + '/cov.npy')	# Get the covariance matrix of proposal distribution from the last run.
	sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior, 
		cov1 , groups=groups1 , outDir=outDir1, verbose=False)
	sampler.sample(x1, 500000, isave=1000)
	chain1 = np.loadtxt(outDir1+'/chain_1.txt')

	# End of the second run.


	# Return the ln-likelihood value of the best fit(maximal likelihood).

	MLHselect = chain1[np.where(chain1==np.max(chain1[:,-3]))[0][0],:]
	Dict = {pta.params[i].name:MLHselect[i] for i in range(ndim)}
	ret.value = (Dict,pta.get_lnlikelihood(Dict),pta.get_lnprior(Dict))


	# End of the function.
#====================================================
if __name__ == '__main__':


	# Importing files.

	datadir = '../ppta/fit1data'
	parfiles = sorted(glob.glob(datadir + '/*.par'))
	timfiles = sorted(glob.glob(datadir + '/*.tim'))

		# I throw away this certain pulsar because it may return a unstable, sometimes
		# very large ln-likelihood value. I think there is a bug somewhere.
	

	psrs = []
	for p, t in zip(parfiles, timfiles):
		psr = Pulsar(p, t)
		psrs.append(psr)

	def get_All_results():
		
		# Modeling each pulsar.

		tm = gp_signals.TimingModel()
		wnb = models.white_noise_block(vary=True)
		dmn = models.dm_noise_block(components=30)
		spn = models.red_noise_block(components=30)
		model = tm + wnb + dmn + spn
		ptas = [signal_base.PTA(model(psr)) for psr in psrs]


		# Multiprocessing.

		jobs = []
		RETs={}	
		for i in range(len(psrs)):	
			RETs[i] = multiprocessing.Manager().Value('i',0)
			p = multiprocessing.Process(target=get_pulsar_noise, args=(ptas[i],RETs[i]))
			jobs.append(p)
			p.start()
		for p in jobs:
			p.join()


		# Return the sum of the Ln Maximal Likelihood values.
		
		MLHselect = [RET.value for RET in RETs.values()]
		return MLHselect


	MLHselect = get_All_results()
		
	MLH = [x[1] for x in MLHselect]
	print sum(MLH)
	 

	# Parameter dictionary, save the best fit noise parameters.

	Dict = {}
	for x in MLHselect:
		Dict.update(x[0])
	save1 = [Dict[key] for key in sorted(Dict.keys())]
	save2 = [key for key in sorted(Dict.keys())]
	np.save('M30/noisepars',save1)
	np.save('M30/noisepardict',save2)







