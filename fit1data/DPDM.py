from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.stats

import enterprise
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
import enterprise.signals.signal_base as base
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
import scipy.constants as sc
import numpy as np

from enterprise_extensions import model_utils

"""
Searching for DPDM signal using the Frequentist method in any PTA data release.

Xiao Xue (2019.09)

xuexiao@mail.itp.ac.cn

"""

@signal_base.function
def dpdm_delay(toas, pos, log10_ma, log10_eps, ra_dp, dec_dp,
                phase_e, dphase):
        
        ma     = 10**log10_ma
        eps    = 10**log10_eps

        c      = sc.c # 299792458.0 [m]/[s]
        A0     = 2.48e-12 * ( 1 / ma ) # [GeV]
        e      = np.sqrt( sc.alpha * 4 * np.pi ) # ~= 0.3028, electric charge in Natural Unit (dimensionless)
        q2m    = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * c **2) * 1e9 # ~= 1/0.938, number of hadrons per GeV, 1/[GeV]
        twopif = 2 * np.pi * ma * sc.electron_volt / sc.h # 2*Pi*f ~= 1.519e+15 * ma (dimensionless)
        rvt2s  = sc.hbar / ( sc.electron_volt ) #  1/([eV]*[s]) 

#        ra  = pos[0]
#        dec = pos[1]

        
#       dx_e = -eps*e*(1/ma)*q2m*A0*np.cos(twopif*toas+phase_e)
#        dx_p = -eps*e*(1/ma)*q2m*A0*np.cos(twopif*toas+phase_e+dphase)
        
        n_dp = [np.cos(dec_dp)*np.cos(ra_dp), np.cos(dec_dp)*np.sin(ra_dp), np.sin(dec_dp)]
        
        ndotn = np.dot(pos, n_dp)       
#        print('omega = ', twopif, "1/[s]")
#        print(twopif*sc.year/2/np.pi, "periods a year")

        Dx = 2.*eps*e*(1/ma)*q2m*A0*np.sin(dphase/2.)*np.sin(twopif*toas + phase_e + dphase/2)

#       delay = rvt2s*ndotn*(dx_p-dx_e)
        delay = rvt2s*ndotn*Dx
        return delay



def dpdm_block(type_,log10_ma=None,log10_eps=None,dec_dp=None,ra_dp=None,phase_e=None,dphase=None):

        # Tips: you may wonder why 'x' before '_dp_', it's beacuse I want to make sure they are the last parameters to show up.
        
        name='x_dp_'
        if log10_ma == None:
                log10_ma  = parameter.Uniform(-23.0, -21.0)(name+'log10_ma')
        if log10_eps == None:
                log10_eps = parameter.Uniform(-28.0, -16.0)(name+'log10_eps')
        if dec_dp == None:
                dec_dp    = parameter.Uniform(-np.pi/2., np.pi/2.)(name+'Dec')
        if ra_dp == None:
                ra_dp     = parameter.Uniform(0 , 2*np.pi)(name+'Ra')
        if phase_e == None:
                phase_e   = parameter.Uniform(0, np.pi)(name+'phase_e')
        
        
        if type_=='Bayes':

                # In Bayes Method, we need to model the DPDM signals on every pulsar at the same time,
                # so that the 'dphase' are included
                if dphase == None:
                        dphase    = parameter.Uniform(-np.pi, np.pi)

                # I don't give 'dphase' a name, so that each pulsar will have its own 'dphase'.
                  
                
                delay = dpdm_delay(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        ra_dp = ra_dp,
                        dec_dp = dec_dp,
                        phase_e = phase_e,
                        dphase = dphase)

        elif type_=='Freq':
                delay = dpdm_delay(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        ra_dp = ra_dp,
                        dec_dp = dec_dp,
                        phase_e = phase_e)


        dpdm = deterministic_signals.Deterministic(delay, name=name)
        return dpdm

def dpdm_block_constant(DP_pars):

        # This block is for single pulsar analysis in the Frequentist scheme, therefore 5 common parameters are constants.

        name = 'x_dp_'
        log10_ma  = parameter.Constant(DP_pars[name+'log10_ma'])(name+'const_log10_ma')
        log10_eps = parameter.Constant(DP_pars[name+'log10_eps'])(name+'const_log10_eps')
        dec_dp    = parameter.Constant(DP_pars[name+'Dec'])(name+'const_Dec')
        ra_dp     = parameter.Constant(DP_pars[name+'Ra'])(name+'const_Ra')
        phase_e   = parameter.Constant(DP_pars[name+'phase_e'])(name+'const_phase_e')
        dphase    = parameter.Uniform(-np.pi, np.pi)(name+'vary_dphase')

        delay = dpdm_delay(log10_ma = log10_ma,
                           log10_eps = log10_eps,
                           ra_dp = ra_dp,
                           dec_dp = dec_dp,
                           phase_e = phase_e,
                           dphase = dphase)

        dpdm = deterministic_signals.Deterministic(delay, name='x_dp')
        return dpdm

def dpdm_test(test,log10_ma=None,log10_eps=None,dec_dp=None,ra_dp=None,phase_e=None,dphase=None):

        @signal_base.function
        def test_delay(toas, pos, log10_ma, log10_eps, ra_dp, dec_dp,
                        phase_e, dphase):
                
                ma     = 10**log10_ma
                eps    = 10**log10_eps

                c      = sc.c
                A0     = 2.48e-12 * ( 1 / ma )
                e      = np.sqrt( sc.alpha * 4 * np.pi )
                q2m    = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * c **2) * 1e9
                twopif = 2 * np.pi * ma * sc.electron_volt / sc.h
                rvt2s  = sc.hbar / ( sc.electron_volt )

                
                n_dp = [np.cos(dec_dp)*np.cos(ra_dp), np.cos(dec_dp)*np.sin(ra_dp), np.sin(dec_dp)]
                
                ndotn = np.dot(pos, n_dp)  
     
                if test == 'earth_only':
                        Dx = eps*e*(1/ma)*q2m*A0*np.cos(twopif*toas+phase_e)
                        delay = rvt2s*ndotn*Dx
                if test == 'psrs_only':
                        Dx = -eps*e*(1/ma)*q2m*A0*np.cos(twopif*toas+phase_e+dphase)
                        delay = rvt2s*ndotn*Dx
                
                return delay
        
        if test == 'earth_only':

                name='x_eo_'

                if log10_ma == None:
                        log10_ma  = parameter.Uniform(-23.0, -21.0)(name+'log10_ma')
                if log10_eps == None:
                        log10_eps = parameter.Uniform(-28.0, -16.0)(name+'log10_eps')
                if dec_dp == None:
                        dec_dp    = parameter.Uniform(-np.pi/2., np.pi/2.)(name+'Dec')
                if ra_dp == None:
                        ra_dp     = parameter.Uniform(0 , 2*np.pi)(name+'Ra')
                if phase_e == None:
                        phase_e   = parameter.Uniform(0, 2*np.pi)(name+'phase_e')

                dphase=parameter.Constant(0)

                delay = test_delay(log10_ma = log10_ma,
                           log10_eps = log10_eps,
                           ra_dp = ra_dp,
                           dec_dp = dec_dp,
                           phase_e = phase_e,
                           dphase = dphase)
                dpdm_test = deterministic_signals.Deterministic(delay, name=name)



        if test == 'psrs_only':

                name='x_po_'

                if log10_ma == None:
                        log10_ma  = parameter.Uniform(-23.0, -21.0)(name+'log10_ma')
                if log10_eps == None:
                        log10_eps = parameter.Uniform(-28.0, -16.0)(name+'log10_eps')
                if dec_dp == None:
                        dec_dp    = parameter.Uniform(-np.pi/2., np.pi/2.)(name+'Dec')
                if ra_dp == None:
                        ra_dp     = parameter.Uniform(0 , 2*np.pi)(name+'Ra')

                phase_e = parameter.Constant(0)(name+'phase_e')

                if phase_e == None:
                        dphase = parameter.Uniform(-np.pi,np.pi)

                delay = test_delay(log10_ma = log10_ma,
                           log10_eps = log10_eps,
                           ra_dp = ra_dp,
                           dec_dp = dec_dp,
                           phase_e = phase_e,
                           dphase = dphase)
                dpdm_test = deterministic_signals.Deterministic(delay, name=name)

        return dpdm_test
        
