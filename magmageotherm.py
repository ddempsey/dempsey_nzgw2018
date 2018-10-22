import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp,cumtrapz
from scipy.optimize import curve_fit
from scipy.stats import expon,lognorm
from ipywidgets import interact, interactive_output, HBox, VBox, IntSlider, Dropdown, RadioButtons, fixed, HTMLMath, Text, Button
from functools import partial
from copy import copy
import json,os

##################
##################    CONSTANTS
##################	
funcs = ['-','deltaP','T','V','eg','em','eX','p','pg','pm','pX','meq','Wgeo','P','Pc','Pcg','M','Mw','H','Mwexsol','Mwdissol','Hsens','Hcrys','Hexsol','Hpv','dpgdP','dpgdT','deXdT','deXdeg','dmeqdP','dmeqdT','dpXdP','dpXdT','dpmdP','dpmdT','dpdP','dpdT','dpdeX','dpdeg','b_1','a','c','dcdP','dcdT','dcdeg','eXin','Min','Mmin','Mwin','Mwout','Mwleak','egW','cin','Hin','Hout','Herupt','Hcool','Hleak','w','dwdt','etar','zbdt','dzs','dPdt','dTdt','degdt','dpmdt','dpXdt','dVdt','Am','Bm','Cm','Dm','Amw','Bmw','Cmw','Dmw','Ah','Bh','Ch','Dh','det','dMbaldt','Mbal','dMwbaldt','Mwbal','dHbaldt','Hbal','Verupt','gamma']
pars = ['b','cm','cX','cg','g','k','A','B','G','Le','Lm','Plith','Tin','Pc','Tl','Ts','TS','am','aX','ar','bm','bX','br','kappa','pr','Mout','Mwin_frac','Pi','Ti','egi','pmi','pXi','Vi','Tbdt','z','dz','Tatm','Merupt','geom','ks','uPc','qfld','label','tmax','eXc']
outputs = ['dV','Ne','egavg','egW','egWavg','Wavg','dVavg','dt','dtavg','dwdtavg','eXavg','dTdtavg']
labels = {
	'-':r'$-$, - []',
	'p':r'$\rho$, mixture density [kg$\,$m$^{-3}$]',
	'P':r'$P$, absolute pressure [MPa]',
	'T':r'$T$, temperature [$^{\circ}$C]',
	'eg':r'$\epsilon_g$, gas fraction []',
	'pm':r'$\rho_m$, melt density [kg$\,$m$^{-3}$]',
	'pX':r'$\rho_X$, crystal density [kg$\,$m$^{-3}$]',
	'V':r'$V$, volume [km$^3$]',
	'M':r'$M$, total mass [kg]',
	'H':r'$H$, total enthalpy [J]',
	'Hsens':r'$H_{sens}$, sensible heat [J]',
	'Hcrys':r'$H_{crys}$, latent heat of crystallisation [J]',
	'Hexsol':r'$H_{exsol}$, latent heat of exsolution [J]',
#	'Hpv':r'$H_{pv}$, pressure-volume work [J]',
	'Mw':r'$M_w$, total water mass [kg]',
	'Mwexsol':r'$M_{w/exsol}$, total exsolved water mass [kg]',
	'Mwdissol':r'$M_{w/dissol}$, total dissolved water mass [kg]',
	'pg':r'$\rho_g$, gas density [kg$\,$m$^{-3}$]',
#	'dpgdP':r'$\partial \rho_g/\partial P$, gas density deriv. wrt. pressure [kg$\,$m$^{-3}$$\,$MPa$^{-1}$]',
#	'dpgdT':r'$\partial\rho_g/\partial T$, gas density deriv. wrt. temperature [kg$\,$m$^{-3}\,$K$^{-1}$]',
	'eX':r'$\epsilon_X$, crystal fraction []',
#	'deXdT':r'$\partial \epsilon_X/\partial T$, crystal fraction deriv. wrt. temperature [K$^{-1}$]',
#	'deXdeg':r'$\partial \epsilon_X/\partial \epsilon_g$, crystal fraction deriv. wrt. gas fraction []',
	'meq':r'$m_{eq}$, water solubility []',
#	'dmeqdP':r'$\partial m_{eq}/\partial P$, water solubility deriv. wrt. pressure [MPa$^{-1}$]',
#	'dmeqdT':r'$\partial m_{eq}/\partial T$, water solubility deriv. wrt. temperature [K$^{-1}$]',
#	'dpXdP':r'$\partial\rho_X/\partial P$, crystal density deriv. wrt. pressure [kg$\,$m$^{-3}$$\,$MPa$^{-1}$]',
#	'dpXdT':r'$\partial\rho_X/\partial T$, crystal density deriv. wrt. te00mperature [kg$\,$m$^{-3}$$\,$K$^{-1}$]',
#	'dpmdP':r'$\partial\rho_m/\partial P$, melt density deriv. wrt. pressure [kg$\,$m$^{-3}$$\,$MPa$^{-1}$]',
#	'dpmdT':r'$\partial\rho_m/\partial T$, melt density deriv. wrt. temperature [kg$\,$m$^{-3}$$\,$K$^{-1}$]',
	'em':r'$\epsilon_m$, melt fraction []',
#	'dpdP':r'$\partial\rho/\partial P$, mixture density deriv. wrt. pressure [kg$\,$m$^{-3}$$\,$MPa$^{-1}$]',
#	'dpdT':r'$\partial\rho/\partial T$, mixture density deriv. wrt. temperature [kg$\,$m$^{-3}$$\,$K$^{-1}$]',
#	'dpdeX':r'$\partial\rho/\partial \epsilon_X$, mixture density deriv. wrt. crystal fraction [kg$\,$m$^{-3}$]',
#	'dpdeg':r'$\partial\rho/\partial \epsilon_g$, mixture density deriv. wrt. gas fraction [kg$\,$m$^{-3}$]',
	'b_1':r'$b_1$, mixture bulk modulus [MPa]',
	'a':r'$a$, mixture thermal expansion coefficient [K$^{-1}$]',
	'c':r'$c$, mixture specific heat [J$\,$kg$^{-1}$K$^{-1}$]',
#	'dcdP':r'$\partial c/\partial P$, mixture specific heat deriv. wrt. pressure [J$\,$kg$^{-1}$K$^{-1}$MPa$^{-1}$]',
#	'dcdT':r'$\partial c/\partial T$, mixture specific heat deriv. wrt. temperature [J$\,$kg$^{-1}$K$^{-1}$K$^{-1}$]',
#	'dcdeg':r'$\partial c/\partial \epsilon_g$, mixture specific heat deriv. wrt. gas fraction [J$\,$kg$^{-1}$K$^{-1}$]',
	'deltaP':r'$\Delta P$, overpressure [MPa]',
	'Pc':r'$\Delta P_c$, critical overpressure (dike) [MPa]',
	'Pcg':r'$\Delta P_g$, critical overpressure (fragmentation) [MPa]',
	'eXin':r'$\epsilon_{X/in}$, recharging crystal fraction []',
	'Min':r'$\dot{M}_{in}$, recharging magma rate [kg$\,$yr$^{-1}$]',
	'Mmin':r'$\dot{M}_{m/in}$, recharging melt rate [kg$\,$yr$^{-1}$]',
	'Mwin':r'$\dot{M}_{w/in}$, recharging water rate [kg$\,$yr$^{-1}$]',
	'Mwout':r'$\dot{M}_{w/out}$, erupting water rate [kg$\,$yr$^{-1}$]',
	'Mwleak':r'$\dot{M}_{w/leak}$, volatile leakage rate [kg$\,$yr$^{-1}$]',
	'egW':r'$\epsilon_{g/geo}$, geothermal volatile fraction []',
	'cin':r'$c_{in}$, recharging specific heat [J$\,$kg$^{-1}$K$^{-1}$]',
	'Hin':r'$\dot{H}_{in}$, recharging enthalpy rate [J$\,$yr$^{-1}$]',
	'Hout':r'$\dot{H}_{out}$, energy loss rate [J$\,$yr$^{-1}$]',
	'Hcool':r'$\dot{H}_{cool}$, energy lost by conduction [J$\,$yr$^{-1}$]',
	'Herupt':r'$\dot{H}_{erupt}$, energy lost by eruption [J$\,$yr$^{-1}$]',
	'Hleak':r'$\dot{H}_{leak}$, energy lost by volatile leakage [J$\,$yr$^{-1}$]',
	'Wgeo':r'$W_{geo}$, geothermal heat flux [W$\,$m$^{-2}$]',
	'Area':r'$A$, area of sill [m$^2$]',
	'w':r'$w$, surface uplift [m]',
	'dwdt':r'$\dot{w}$, uplift rate [mm$\,$yr$^{-1}$]',
	'etar':r'$\eta_r$, viscosity of viscoelastic shell [Pa$\,$s]',
	'zbdt':r'$z_{BDT}$, depth to BDT [m]',
	'dzs':r'$z_{BDT}-z_{mb}$, thickness of viscoelastic shell [m]',
	'Verupt':r'$V_{erupt}$, cumulative erupted volume [km$^3$]',
	'gamma':r'$\gamma$, geothermal power multiplier',
#	'dPdt':r'$dP/dt$, pressure deriv. wrt. time [MPa$\,$yr$^{-1}$]',
#	'dTdt':r'$dT/dt$, temperature deriv. wrt. time [K$\,$yr$^{-1}$]',
#	'degdt':r'$d\epsilon_g/dt$, gas fraction deriv. wrt. time [yr$^{-1}$]',
#	'dpmdt':r'$d\rho_m/dt$, melt density deriv. wrt. time [kg$\,$m$^{-3}$$\,$yr$^{-1}$]',
#	'dpXdt':r'$d\rho_X/dt$, crystal density deriv. wrt. time [kg$\,$m$^{-3}$$\,$yr$^{-1}$]',
	'dVdt':r'$dV/dt$, volume deriv. wrt. time [km$^3\,$yr$^{-1}$]',
#	'Am':r'$A_m$, Am []',
#	'Bm':r'$B_m$, Bm []',
#	'Cm':r'$C_m$, Cm []',
#	'Dm':r'$D_m$, Dm []',
#	'Amw':r'$A_{mw}$, Amw []',
#	'Bmw':r'$B_{mw}$, Bmw []',
#	'Cmw':r'$C_{mw}$, Cmw []',
#	'Dmw':r'$D_{mw}$, Dmw []',
#	'Ah':r'$A_h$, Ah []',
#	'Bh':r'$B_h$, Bh []',
#	'Ch':r'$C_h$, Ch []',
#	'Dh':r'$D_h$, Dh []',
#	'det':r'$\Delta$, det []',
#	'dMbaldt':r'$A_m\dot{P}+B_m\dot{T}+C_m\dot{\epsilon}_g+D_m$, mass balance error [kg/yr]',
#	'Mbal':r'$\Delta M$, total mass error [kg]',
#	'dMwbaldt':r'$A_{mw}\dot{P}+B_{mw}\dot{T}+C_{mw}\dot{\epsilon}_g+D_{mw}$, water balance error [kg/yr]',
#	'Mwbal':r'$\Delta M_w$, total water error [kg]',
#	'dHbaldt':r'$A_h\dot{P}+B_h\dot{T}+C_h\dot{\epsilon}_g+D_h$, enthalpy balance error [J/yr]',
#	'Hbal':r'$\Delta H$, total enthalpy error [J]'
	}
dropdown_options = dict([(labels[k].split(',')[1].strip().split('[')[0].strip(),k) for k in funcs if k in labels.keys()])
# coefficients for computing density
YDL=[ 0.10000000e+01, 0.17472599e-01, -0.20443098e-04, -0.17442012e-06, 0.49564109e-02, -0.40757664e-04,0.50676664e-07, 0.50330978e-04, 0.33914814e-06, -0.18383009e-06,] 
ZDL=[ 0.10009476e-02, 0.16812589e-04, -0.24582622e-07, -0.17014984e-09, 0.48841156e-05, -0.32967985e-07, 0.28619380e-10, 0.53249055e-07, 0.30456698e-09, -0.12221899e-09,]
# scaling
YR_SECS = 365.25*24*3600
scales = (
[['P','dPdt','deltaP','b_1','Pc','Pcg','uPc','Pi'],1.e-6],
[['dpgdP','dmeqdP','dpXdP','dpmdP','dpdP','dcdP'],1.e6],
[['t','dt','dtavg'],1./YR_SECS],
[['dPdt','dTdt','dTdtavg','degdt','dpmdt','dpXdt','dVdt','dMbaldt','dMwbaldt','dHbaldt','Min','Mmin','Mwin','Mwout','Mwleak','cin','Hin','Hout','Hcool','Hleak','Herupt','dwdt','dwdtavg'],YR_SECS],
[['V','dVdt','dV','dVavg','Vi','Verupt'],1.e-9],
[['dwdt','dwdtavg'],1.e3],
[['dTdtavg'],1.e3],
)		


class SimulationError(Exception):
	pass
save_attrs = funcs + pars + outputs + ['t','te','tmax','terminate_string']
save_attrs.remove('-')
save_attrs.remove('Min')
save_attrs.remove('gamma')
mindist_attrs = ['avgdt','ndt','dvavg','dvscale','dtdv','dvmax','dvshape','seed','constant','Min_constant']
gammadist_attrs = ['base','freq','amp','phase','constant','gamma_constant']

##################
##################    MODEL OBJECT
##################	
class MagmaChamber(object):
	def __init__(self, **kwargs):
		# set parameters
		self.set_defaults()
		# overwrite default initial conditions
		ivars = ['Pi','Ti','egi','pmi','pXi','Vi']
		pars = [k for k in kwargs.keys() if k not in ivars]
		for key in pars:
			self.__setattr__(key, kwargs[key])
		self.set_initial_conditions()
		# overwrite default parameters
		vars = [k for k in kwargs.keys() if k in ivars]
		for key in vars:
			self.__setattr__(key, kwargs[key])
			
		# edge case 1: set constant rock viscosity
		if 'etar' in kwargs.keys():
			self._etari = kwargs['etar']
			
		# edge case 2: constant or user specified mass input
		if 'Min' in kwargs.keys():
			if type(kwargs['Min']) is list:
				self.Min = MinDist(None,None,None,None,None,vector=kwargs['Min'])
			elif type(kwargs['Min']) in [float, int]:
				self.Min = MinDist(None,None,None,None,None,constant=kwargs['Min'])
			
		# edge case 3: constant gamma
		if 'gamma' in kwargs.keys():
			if type(kwargs['gamma']) is not GammaDist:
				self.gamma = GammaDist(None,None,None,None,constant=kwargs['gamma'])
		# edge case 4: dry magma, reassign volatile functions
		if 'dry' in kwargs.keys():
			if kwargs['dry']:
				self.pgf = lambda : 0.
				self.dpgdTf = lambda : 0.
				self.dpgdPf = lambda : 0.
				self.meqf = lambda : 0.
				self.dmeqdTf = lambda : 0.
				self.dmeqdPf = lambda : 0.
				self.Le = 0.
				self.egi = 0.
				self.degdtf = lambda : 0.
				self.Mwleakf = lambda : 0.
				self.Pcgf = lambda : 1.e32
	def __repr__(self):
		return 'MagmaChamber'
	def set_defaults(self):				# sets default parameters
		# material parameters
		self.am = 1.e-5
		self.ar = 1.e-5
		self.aX = 1.e-5
		self.A = 1.e9						# Dorn parameter (del Negro 2009)
		self.b = 0.5
		self.bm = 1.e10
		self.br = 1.e10
		self.bX = 1.e10
		self.B = 8.31						# Gas constant
		self.cm = 1205.
		self.cX = 1315.
		self.cg = 3900.
		self.dz	= 1.e3						# thickness		
		self.etari = None
		self.etag = 5.2e-5					# viscosity water at 800degC and 100 MPa
		self.erupting = False
		self.eXc = 0.5
		self.g = 9.81
		self.gamma = GammaDist(None,None,None,None,constant=10.)
		self.geom = 1
		self.G = 120.e3						# activation energy (del Negro 2009)
		self.k = 2.5
		self.kappa = 1.e-6
		self.ks = 1.e-18 					# permeability of viscoelastic shell
		self.Le = 61.e4						# exsolution
		self.Lm = 29.e4						# melting
		self.Lpv = 0.						# binary, to switch off enthalpy effects
		self.Mout = 0.
		self.Min = MinDist(None,None,None,None,None,constant=1.)
		self.Merupt = 1.e4
		self.Mwin_frac = 0.05
		self.qfld = 1.2/(365.25*24*3600)*0.02*1.e3			# mass flux of overlying geothermal circulation (rainfall x infiltration x density)
		self.pr = 2600.
		self.Plith = 2.e8
		self.Pc = 2.e7
		self.tmax = 1.6e11
		self.Tatm = 25.+273.				# surface temperature 
		self.Tbdt = 350.+273.				# BDT transition temperature
		self.Tin = 1200
		self.Tl = 1223
		self.Ts = 973
		self.TS = 500
		self.uPc = 0.
		self.z = 5.e3						# burial depth	
		# outputs
		self.dry = False
		self.dwdtavg = None
		self.eXavg = None
		self.dTdtavg = None
		self.label = ''
		# simulation parameters
		self.simulation_complete = False
		self.Ne_max = 20
		self.rtol = 1.e-12
	def set_initial_conditions(self):	# sets default initial conditions
		# default initial conditions	
		self.Pi = 1.*self.Plith			# pressure is lithostatic
		self.Ti = 1200.
		self.egi = 0.04
		self.pmi = 2400.
		self.pXi = 2600.
		self.Vi = 3.e6
	def run(self, Tupdates=100, verbose=False):			# runs the simulation
		#self.tmax = tmax*YR_SECS	
		
		# set up variable if Min
		self.Min.sample(self.tmax)
		
		# save simulation max time and compute temperature update step
		self.i_s = 0
		self.t_s = np.linspace(0,self.tmax,Tupdates+1)
		self.dt_s = self.t_s[1]-self.t_s[0]
		self.m_s = 0.*self.t_s
		self.T_s = 0.*self.t_s; self.T_s[self.i_s] = self.Ti
		self.p_s = 0.*self.t_s
		self.verbose = verbose
		
		# termination events - eruption and dormancy
		erupt_event1 = lambda t,y: eruption_condition1(y,t,self)
		erupt_event1.terminal = True
		erupt_event1.direction = 1
		erupt_event2 = lambda t,y: eruption_condition2(y,t,self)
		erupt_event2.terminal = True
		erupt_event2.direction = 1
		dormancy_event1 = lambda t,y: dormancy_condition1(y,t,self)
		dormancy_event1.terminal = True
		dormancy_event2 = lambda t,y: dormancy_condition2(y,t,self)
		dormancy_event2.terminal = True
		
		# termination events - out of bounds
			# melting curve validity
		tc1 = lambda t,y: y[1]-self.Ts
		tc1.terminal = True
		tc2 = lambda t,y: y[1]-self.Tl
		tc2.terminal = True
			# gas density validity
		tc3 = lambda t,y: y[1]-873.
		tc3.terminal = True
		tc4 = lambda t,y: 1.#y[1]-1173.
		tc4.terminal = True
		tc5 = lambda t,y: y[0]-30.e6
		tc5.terminal = True
		tc6 = lambda t,y: y[0]-400.e6
		tc6.terminal = True
			# gas fraction
		tc7 = lambda t,y: y[2]-1.
		tc7.terminal = True
		tc8 = lambda t,y: y[2]
		tc8.terminal = True
			# crystal fraction
		tc9 = lambda t,y: (1-y[2])*(1-((y[1]-self.Ts)/(self.Tl-self.Ts))**self.b)-(0.99-y[2])
		tc9.terminal = True
		tc10 = lambda t,y: (1-y[2])*(1-((y[1]-self.Ts)/(self.Tl-self.Ts))**self.b)
		tc10.terminal = True
			# melt fraction
		tc11 = lambda t,y: (1.-y[2]-(1-y[2])*(1-((y[1]-self.Ts)/(self.Tl-self.Ts))**self.b))-1.
		tc11.terminal = True
		tc12 = lambda t,y: (1.-y[2]-(1-y[2])*(1-((y[1]-self.Ts)/(self.Tl-self.Ts))**self.b))
		tc12.terminal = True
		tc_events = [tc1,tc2,tc3,tc4,tc5,tc6,tc7,tc8,tc9,tc10,tc11,tc12]
		if self.dry:
			for i in [2,3,4,5,6,7]:
				tc_events[i] = lambda t,y: 1.
		event_labels = [r'\(T<T_s\)',r'\(T>T_l\)',r'\(T<600^{\circ}C\)',r'\(T>900^{\circ}C\)',r'\(P<30\)MPa',r'\(P>400\)MPa',r'\(\epsilon_g>\)1',r'\(\epsilon_g<\)0',r'magma has solidified \(\epsilon_m=0\)',r'\(\epsilon_X<\)0',r'\(\epsilon_m>\)1',r'\(\epsilon_m<\)0']
		
		# initialise simulation in dormancy
		Nout = 1001
		t = np.linspace(0,self.tmax,Nout)
		self.te = [t[0],]
		keep_simulating = True
		erupting = False
		self.Mout = 0.
		events = tc_events + [erupt_event1,erupt_event2]
		y0 = [self.Pi,self.Ti,self.egi,self.pmi,self.pXi,self.Vi]
		self.tv = np.array([])
		self.yv = np.array([[],[],[],[],[],[]])
		while keep_simulating:
			# run subsimulation
			sufficient_output = False
			
			attempts = 0
			while not sufficient_output:
				
				out = solve_ivp(fun = lambda t,y: dxdt(y,t,self), t_span = [t[0],t[-1]], y0 = y0, t_eval = t, method='Radau', rtol = self.rtol, atol=1.e-12, events = events)
				
				if out.status == -1:
				# simulation did not finish
					if erupting:
						raise SimulationError('solver failed while erupting')
						# failing on eruption, try to get some output for diagnostic
						t = np.linspace(t[0],82948727717.11264,41)
						out = solve_ivp(fun = lambda t,y: dxdt(y,t,self), t_span = [t[0],t[-1]], y0 = y0, t_eval = t, method='RK45', rtol = self.rtol, atol=1.e-12, events = events)
						if out.status == 0:
							keep_simulating = False
							self.terminate_string = 'solver failed while erupting'
							break
					else:
						#raise SimulationError('solver failed while erupting')
						keep_simulating = False
						self.terminate_string = 'solver failed while dormant'
						break
				
				elif out.status == 0:
				# simulation finished
					if len(out.t)>20:
						sufficient_output = True
						continue
				
				elif out.status == 1:
				# simulation halted at event
				
					# check case that event time occurs before initial time
					if abs(get_event_time(out.t_events)-t[0])/(1+t[0]) <= self.rtol:
						keep_simulating = False
						if erupting:
							event_type = 'eruption'
						else:
							event_type = 'dormancy'
						self.terminate_string = 'solver failed to resolve {} event'.format(event_type)
						break
				
				# completed the simulation but not enough output, restart
				if len(out.t)>20:
					sufficient_output = True
				else:
					tend = get_event_time(out.t_events)
					tstart = self.te[-1]
					tend = tstart + 1.2*(tend-tstart)
					t = np.linspace(tstart,tend,41)
				
				# allow a few attempts to get the right number of outputs
				attempts += 1
				if attempts == 3:
					raise SimulationError
			
			# check reason for finishing			
			if out.status == 1:
				# event occurred
				self.te.append(get_event_time(out.t_events))
				if len(out.t_events[-1])==0 and len(out.t_events[-2])==0:
					# out of bounds event
					keep_simulating = False
					for i, te in enumerate(out.t_events):
						if len(te)>0:
							self.terminate_string = 'out of bounds - '+event_labels[i]
							break
				else:
					# eruption or dormancy event
					erupting = not erupting
					if erupting:
						self.Mout = 1.*self.Merupt
						events = tc_events + [dormancy_event1, dormancy_event2]
						t = np.linspace(self.te[-1],self.tmax,np.max([Nout-len(self.tv),21])) 
					else:
						self.Mout = 0.
						events = tc_events + [erupt_event1, erupt_event2]
						t = np.linspace(self.te[-1],self.tmax,np.max([Nout-len(self.tv),21])) 
						
					y0 = out.y[:,-1]
			
			elif out.status == 0:
				# end of simulation
				self.te.append(out.t[-1])
				self.terminate_string = 'simulation completed, {:d} eruptions'.format((len(self.te)-1)//2)
				keep_simulating = False
			
			# check max eruptions not exceeded
			if (len(self.te)-1)//2 == self.Ne_max:
				keep_simulating = False
				self.terminate_string = 'maximum number of eruptions ({:d}) reached before simulation completed'.format((len(self.te)-1)//2)
			
			# save simulation output
			self.tv = np.concatenate([self.tv,out.t])
			self.yv = np.concatenate([self.yv,out.y],axis=1)
			
			# check exit condition
			if self.tv[-1] >= self.tmax:
				keep_simulating = False
				self.terminate_string = 'simulation completed, {:d} eruptions'.format((len(self.te)-1)//2)
		
		self.x = self.yv
		self.t = self.tv
		self.post_process()
		self.simulation_complete=True
	def post_process(self):				# runs post-processing routines
		# compute variables for duration of simulation
		self.set_eruption_rate()
		dxdt(self.yv,self.tv,self)
		
		# primary variables
		self.P = self.Pf()										# pressure
		self.T = self.Tf()										# temperature
		self.eg = self.egf()									# water fraction
		self.pm = self.pmf()									# melt density
		self.pX = self.pXf()									# crystal density
		self.V = self.Vf()										# volume
		
		# compute balances
		self.M = self.Mf()										# total mass
		self.H = self.Hf()										# total enthalpy
		self.Hsens = self.Hsensf()								# sensible heat
		self.Hcrys = self.Hcrysf()								# heat of crystallisation
		self.Hexsol = self.Hexsolf()							# heat of exsolution
		self.Hpv = self.Hpvf()									# volume change
		self.Mw = self.Mwf()									# total water
		self.Mwexsol = self.Mwexsolf()							# exsovled water
		self.Mwdissol = self.Mwdissolf()						# dissolved water
		self.dMbaldt = self.dMbaldtf()*self.p*self.V    		# instantaneous mass balance
		self.Mbal = self.Mbalf()								# cumulative mass balance
		self.dMwbaldt = self.dMwbaldtf()*self.V           		# instantaneous water balance
		self.Mwbal = self.Mwbalf()								# cumulative water balance
		self.dHbaldt = self.dHbaldtf()*self.T*self.V*self.cxp   # instantaneous enthalpy balance
		self.Hbal = self.Hbalf()								# cumulative enthalpy balance
		
		# miscellaneous
		self.Verupt = self.Veruptf() 							# cumulative erupted volume
		
		# perform analysis on simulation
		self.analyse()
	def analyse(self):					# analyse the simulation output
		# compute uplift quantities
		self.w = self.wf()
		self.dwdt = self.dwdtf()
		
		# number of eruptions
		self.Ne = (len(self.te)-1)//2
		
		# average time between eruptions (exclude initial)
		if self.Ne > 1:
			self.dt = np.array([np.diff(self.te[2*i:2*(i+1)])[0] for i in range(1,self.Ne)])
			self.dtavg = np.mean(self.dt)
		else:
			self.dtavg = None
		
		# average erupted volume
		if self.Ne > 0:
			dVs = []
			for i in range(1,self.Ne+1):
				t0,t1 = self.te[2*i-1:2*i+1]
				i0 = np.argmin(abs(self.t - t0))
				i1 = np.argmin(abs(self.t - t1))
				dVs.append(self.Verupt[i1]-self.Verupt[i0])
			self.dV = np.array(dVs)
			self.dVavg = np.mean(self.dV)
		else:
			self.dVavg = None
		
		# average volatiles in geothermal
		if self.Ne > 1:
			t0 = self.te[2]
			i0 = np.argmin(abs(self.t - t0))
			t1 = self.te[2*self.Ne-1]
			i1 = np.argmin(abs(self.t - t1))
		else:
			i0 = 0
			i1 = -1
		if not self.dry:
			self.egW = self.egWf()
		else:
			self.egW = 0.*self.t
		self.egWavg = np.trapz(self.egW[i0:i1],self.t[i0:i1])/(self.t[i1]-self.t[i0])
		
		# average uplift rate during inter-eruption
		if self.Ne > 1:
			dwdt = 0.
			dt = 0.
			for i in range(1,self.Ne):
				t2,t3 = self.te[2*i:2*i+2]
				i2 = np.argmin(abs(self.t - t2))
				i3 = np.argmin(abs(self.t - t3))
				dwdt += np.trapz(self.dwdt[i2+1:i3],self.t[i2+1:i3])
				dt += self.t[i3-1]-self.t[i2+1]
			self.dwdtavg = dwdt/dt
		else:
			self.dwdtavg = None
		
		# average geothermal heat flux
		if not self.dry:
			self.Wavg = np.trapz(self.Wgeo[i0:i1],self.t[i0:i1])/(self.t[i1]-self.t[i0])
		else:
			self.Wavg = None
		
		# average crystal fraction at eruption
		if self.Ne > 0:
			self.eXavg = np.mean([self.eX[np.argmin(abs(self.t - self.te[2*i-1]))] for i in range(1,self.Ne+1)])
		else:
			self.eXavg = None
		
		# average volatile fraction at eruption
		if self.Ne > 0 and not self.dry:
			self.egavg = np.mean([self.eg[np.argmin(abs(self.t - self.te[2*i-1]))] for i in range(1,self.Ne+1)])
		else:
			self.egavg = None
		
		# average cooling/heating rate
		try:
			p,pcov = curve_fit(lambda t,p1,p2: p1*t+p2, self.t[i0:i1], self.T[i0:i1], [(self.T[-1]-self.T[0])/self.t[-1],self.Ti])
			self.dTdtavg = -p[0]
		except:
			self.dTdtavg = None
	def get(self,vars,plot_options=''):	# gets simulation output for plotting
		ys = []
		for var in vars:
			if var == '-':
				y = self.t
			elif var == 'Min':
				y = self.Min(self.t)
			elif var == 'gamma':
				y = self.gamma(self.t)
			else:
				y = self.__getattribute__(var)
			ys.append(self.transform(y,var))
		ys = np.array(ys).T
		
		t = self.transform(self.t,'t')
		
		# truncate results if requested
		n = len(self.te)
		if plot_options == 'latest dormancy':
			t0,t1 = self.te[n-2-n%2:n-n%2]
		elif plot_options == 'latest eruption':
			try: 
				t0,t1 = self.te[n-2-(n-1)%2:n-(n-1)%2]
			except ValueError:
				raise SimulationError('no eruption occurred in this simulation')
		else:
			t0,t1 = [self.te[0], self.te[-1]]
			
		inds = np.where((self.t>=t0)&(self.t<t1))
		return t[inds], ys[inds,:].T
	def load(self,savefile):			# loads a previous simulation
		with open(savefile) as fp:
			savedict = json.loads(fp.read())
		self.Min = MinDist(None,None,None,None,None)
		self.gamma = GammaDist(savedict['gammadist_base'],savedict['gammadist_freq'],savedict['gammadist_amp'],constant=savedict['gammadist_gamma_constant'])
		for k in savedict.keys():
			val = savedict[k]
			try:
				_ = (xi for xi in val)
				val = np.array(val)
			except TypeError:
				pass
			if k.startswith('mindist_'):
				self.Min.__setattr__(k.replace('mindist_',''), val)
			elif k.startswith('gammadist_'):
				self.gamma.__setattr__(k.replace('gammadist_',''), val)
			else:
				self.__setattr__(k,val)
		self.Min.sample(self.tmax)
		self.terminate_string = str(self.terminate_string)
		self.label = str(self.label)
	def save(self,savefile='out.mg'):	# saves the simulation
		if not self.simulation_complete:
			raise SimulationError('Run model before saving.')
		# list of attributes to save
		savedict = dict([(attr, self.__getattribute__(attr)) for attr in save_attrs])
		mindistdict = dict([('mindist_'+attr, self.Min.__getattribute__(attr)) for attr in mindist_attrs])
		savedict.update(mindistdict) 
		gammadistdict = dict([('gammadist_'+attr, self.gamma.__getattribute__(attr)) for attr in gammadist_attrs])
		savedict.update(gammadistdict) 
		jsonmc = JSONMC(**savedict)
		with open(savefile,'w') as fp:
			fp.write(json.dumps(jsonmc.__dict__, sort_keys=True, indent=4, separators=(',', ': ')))
	
	
	##################
	##################    THERMODYNAMIC FUNCTIONS
	##################	
	def pgf(self):						# gas density 
		P,T,eg,pm,pX,V = self.x
		P2 = P/1.e5
		T2 = T-273.
		return 1.e3*(-112.528*T2**-0.381+127.811*P2**-1.135+112.04*T2**-0.411*P2**0.033)
	def dpgdPf(self):					# gas density derivative wrt pressure
		P,T,eg,pm,pX,V = self.x
		P2 = P/1.e5
		T2 = T-273.
		return 1.e3*(-145.065485*P2**(-2.135) + 3.69732*P2**(-0.967)*T2**(-0.411))/1.e5
	def dpgdTf(self):					# gas density derivative wrt temperature
		P,T,eg,pm,pX,V = self.x
		P2 = P/1.e5
		T2 = T-273.
		return 1.e3*(-46.04844*P2**0.033*T2**(-1.411) + 42.873168*T2**(-1.381))
	def eXf(self):						# crystal fraction (melting curve)
		P,T,eg,pm,pX,V = self.x
		try: 
			_ = (xi for xi in T)
			eX = 1.-eg
			inds = np.where(T>self.Ts)
			eX[inds] = (1-eg[inds])*(1-((T[inds]-self.Ts)/(self.Tl-self.Ts))**self.b)
			inds = np.where(T>self.Tl)
			eX[inds] = 0.
		except TypeError:
			if T < self.Ts:
				eX = (1.-eg)
			elif T > self.Tl:
				eX = 0.
			else:
				eX = (1-eg)*(1-((T-self.Ts)/(self.Tl-self.Ts))**self.b)
		return eX
	def deXdTf(self):					# crystal fraction derivative wrt to temperature
		P,T,eg,pm,pX,V = self.x
		try: 
			_ = (xi for xi in T)
			deXdT = 0.*T
			inds = np.where(T>self.Ts)
			deXdT[inds] = (1.-eg[inds])*-self.b*((T[inds]-self.Ts)**(self.b-1)/(self.Tl-self.Ts)**self.b)
			inds = np.where(T>self.Tl)
			deXdT[inds] = 0.
		except TypeError:
			if T <= self.Ts:
				deXdT = 0.
			elif T > self.Tl:
				deXdT = 0.
			else:
				deXdT = (1.-eg)*-self.b*((T-self.Ts)**(self.b-1)/(self.Tl-self.Ts)**self.b)
		return deXdT
	def deXdegf(self):					# crystal fraction derivative wrt to gas fraction
		P,T,eg,pm,pX,V = self.x
		try: 
			_ = (xi for xi in T)
			deXdeg = -1.+0.*T
			inds = np.where(T>self.Ts)
			deXdeg[inds] = -(1-((T[inds]-self.Ts)/(self.Tl-self.Ts))**self.b)
			inds = np.where(T>self.Tl)
			deXdeg[inds] = 0.
		except TypeError:
			if T < self.Ts:
				deXdeg = -1.
			elif T > self.Tl:
				deXdeg = 0.
			else:
				deXdeg = -(1-((T-self.Ts)/(self.Tl-self.Ts))**self.b)
		return deXdeg
	def meqf(self):						# dissolved water content (solubility curve)
		P,T,eg,pm,pX,V = self.x
		P2 = P/1.e6
		return (P2**0.5*(0.4874-608/T+489530/T**2)+P2*(-0.06062 +135.6/T - 69200/T**2)+P2**1.5*(0.00253-4.154/T+1509/T**2))/100.
	def dmeqdPf(self):					# dissolved water content derivative wrt to pressure
		P,T,eg,pm,pX,V = self.x
		P2 = P/1.e6
		#return (0.005*P**(-0.5)*(0.4874 - 608/T + 489530/T**2) + 0.015*P**0.5*(0.00253 - 4.154/T + 1509/T**2) - 0.0006062 + 1.356/T - 692.0/T**2)
		return ((0.5*P2**(-0.5)*(0.4874 - 608/T + 489530/T**2) + 1.5*P2**0.5*(0.00253 - 4.154/T + 1509/T**2) - 0.06062 + 135.6/T - 69200/T**2)/100.)/1.e6
	def dmeqdTf(self):					# dissolved water content derivative wrt to temperature
		P,T,eg,pm,pX,V = self.x
		P2 = P/1.e6
		#return 0.01*P**0.5*(608/T**2 - 979060/T**3) + 0.01*P*(-135.6/T**2 + 138400/T**3) + 0.01*P**1.5*(4.154/T**2 - 3018/T**3)
		return (P2**0.5*(608/T**2 - 979060/T**3) + P2*(-135.6/T**2 + 138400/T**3) + P2**1.5*(4.154/T**2 - 3018/T**3))/100.
	def dpXdPf(self):					# crystal density derivative wrt pressure  (compressibility)
		P,T,eg,pm,pX,V = self.x
		return pX/self.bX
	def dpXdTf(self):					# crystal density derivative wrt temperature  (thermal expansion)
		P,T,eg,pm,pX,V = self.x
		return -pX*self.aX
	def dpmdPf(self):					# melt density derivative wrt pressure  (compressibility)
		P,T,eg,pm,pX,V = self.x
		return pm/self.bm
	def dpmdTf(self):					# melt density derivative wrt temperature  (thermal expansion)
		P,T,eg,pm,pX,V = self.x
		return -pm*self.am
	def emf(self):						# melt fraction
		P,T,eg,pm,pX,V = self.x
		return 1.-eg-self.eX
	def pf(self):						# mixture density
		P,T,eg,pm,pX,V = self.x
		return self.em*pm+self.eX*pX+eg*self.pg
	def dpdPf(self):					# mixture density derivative wrt pressure
		P,T,eg,pm,pX,V = self.x
		return self.eX*self.dpXdP + eg*self.dpgdP + self.dpmdP*self.em
	def dpdTf(self):					# mixture density derivative wrt temperature
		P,T,eg,pm,pX,V = self.x
		return self.deXdT*(pX-pm)+eg*self.dpgdT+self.eX*self.dpXdT +self.em*self.dpmdT
	def dpdeXf(self):					# mixture density derivative wrt crystal fraction
		P,T,eg,pm,pX,V = self.x
		return pX-pm
	def dpdegf(self):					# mixture density derivative wrt gas fraction
		P,T,eg,pm,pX,V = self.x
		return self.deXdeg*(pX-pm)+self.pg-pm
	def b_1f(self):						# mixture bulk modulus
		P,T,eg,pm,pX,V = self.x
		return (self.em*pm/self.bm+self.eX*pX/self.bX+eg*self.dpgdP)/self.p
	def af(self):						# mixture thermal expansion
		P,T,eg,pm,pX,V = self.x
		return (self.em*pm*self.am+self.eX*pX*self.aX-eg*self.dpgdT)/self.p
	def cxpf(self):						# mixture: product of specific heat and density
		P,T,eg,pm,pX,V = self.x
		return (pX*self.eX*self.cX+self.pg*eg*self.cg+pm*self.em*self.cm)  
	def cf(self):						# mixture specific heat
		P,T,eg,pm,pX,V = self.x
		return self.cxp/self.p
	def dcdPf(self):					# mixture specific heat derivative wrt pressure
		P,T,eg,pm,pX,V = self.x
		return (self.dpXdP*self.eX*self.cX+self.dpgdP*eg*self.cg+self.dpmdP*self.em*self.cm)/self.p-self.c/self.p*self.dpdP
	def dcdTf(self):					# mixture specific heat derivative wrt temperature
		P,T,eg,pm,pX,V = self.x
		return (self.dpXdT*self.eX*self.cX+self.dpgdT*eg*self.cg+self.dpmdT*self.em*self.cm)/self.p-self.c/self.p*self.dpdT+((pX*self.cX-pm*self.cm)/self.p-self.c/self.p*self.dpdeX)*self.deXdT
	def dcdegf(self):					# mixture specific heat derivative wrt gas fraction
		P,T,eg,pm,pX,V = self.x
		#return ((pX*self.cX-pm*self.cm)*self.deXdeg+self.pg*self.cg-pm*self.cm)/self.p-self.c/self.p*self.dpdeg-self.c/self.p*self.dpdeX*self.deXdeg
		return (self.pg*self.cg-pm*self.cm - self.c*self.dpdeg +(pX*self.cX-pm*self.cm - self.c*self.dpdeX)*self.deXdeg)/self.p
	def deltaPf(self):					# chamber overpressure
		P,T,eg,pm,pX,V = self.x
		return P - self.Plith
	def eXinf(self):					# recharging crystal fraction
		P,T,eg,pm,pX,V = self.x
		if self.Tin<self.Ts:
			raise SimulationError('recharging magma is solid!')
		elif self.Tin>self.Tl:
			return 0.
		else:
			return (1-((self.Tin-self.Ts)/(self.Tl-self.Ts))**self.b)
	def Mminf(self):					# recharging melt mass
		P,T,eg,pm,pX,V = self.x
		return self.Min(self.t)*(1.-self.eXin)    
	def Mwinf(self):					# recharging water mass
		P,T,eg,pm,pX,V = self.x
		return self.Mwin_frac*self.Mmin
	def Mwoutf(self):					# erupting water mass
		P,T,eg,pm,pX,V = self.x
		self.Mwleak = self.Mwleakf()
		return self.Mout*(self.pg*eg+self.meq*pm*self.em)/self.p + self.Mwleak
	def Mwleakf(self):					# leakage rate across shell
		P,T,eg,pm,pX,V = self.x
		dPdz = (P-self.Pbdt)*((P-self.Pbdt)/(2*self.bg*self.pg)-1)/self.dzs
		return self.Area*self.ks*self.pg/self.etag*-dPdz*eg
	def cinf(self):						# recharging mixture specific heat
		P,T,eg,pm,pX,V = self.x
		return (pX*self.eX*self.cX + (1.-self.eX)*pm*self.cm)/self.p
	def Hinf(self):						# recharging enthalpy
		P,T,eg,pm,pX,V = self.x
		return self.cin*self.Tin*self.Min(self.t)
	def Houtf(self):					# erupting enthalpy
		P,T,eg,pm,pX,V = self.x
		self.Hleak = self.Hleakf()
		self.Wgeo = self.Wgeof()
		self.Herupt = self.Heruptf()
		return self.Herupt + self.Hcool + self.Hleak
	def zbdtf(self):					# BDT depth
		P,T,eg,pm,pX,V = self.x
		return self.z-(T-self.Tbdt)*self.z/((1.+self.gamma(self.t))*(self.Tbdt-self.Tatm)+(T-self.Tbdt))
	def dzsf(self):						# thickness of viscoelastic shell
		P,T,eg,pm,pX,V = self.x
		return self.z - self.zbdt
	def Pbdtf(self):					# cold hydrostatic pressure at BDT
		P,T,eg,pm,pX,V = self.x
		return self.zbdt*1.e3*9.81
	def bgf(self):						# water compressibility through shell
		P,T,eg,pm,pX,V = self.x
		pgbdt = dens(self.Pbdt/1.e6, self.Tbdt-273.)
		return (self.Pbdt-P)/(pgbdt-self.pg)
	def Areaf(self):					# area of sill
		P,T,eg,pm,pX,V = self.x
		if self.geom == 1:
			return self.Vi/self.dz
		else:
			return V/self.dz
	def wf(self):						# surface uplift
		P,T,eg,pm,pX,V = self.x
		return (V-self.Vi)/self.Area
	def dwdtf(self):					# surface uplift rate
		P,T,eg,pm,pX,V = self.x
		dwdt = self.dVdt/self.Area
		erupting = False
		i = 0
		for tii in self.te[1:]:
			while self.t[i] <= tii and i<len(self.t)-1:
				if erupting:
					dwdt[i] = float('NaN')
				i += 1
				if i == len(self.t)-1:
					return dwdt
			erupting = not erupting	
	def Heruptf(self):					# enthalpy lost during eruption
		P,T,eg,pm,pX,V = self.x
		return self.c*T*self.Mout
	def Hcoolf(self):					# geothermal cooling rate of sill
		P,T,eg,pm,pX,V = self.x
		C = -self.cg*self.ks*(P-self.Pbdt)*((P-self.Pbdt)-2*self.bg*self.pg)/(2*self.k*self.etag*self.bg*self.dzs)
		dTdz = -C*(T-self.Tbdt)/(1-np.exp(-C*self.dzs))
		return self.Area*self.k*-dTdz
	def Hleakf(self):					# leakage rate through shell
		P,T,eg,pm,pX,V = self.x
		return self.Mwleak*self.cg*T
	def egWf(self):						# volatile fraction in geothermal mass flux
		P,T,eg,pm,pX,V = self.x
		return self.Mwleak/self.Area/(self.qfld+self.Mwleak/self.Area)
	def Wgeof(self):					# geothermal heat flux
		P,T,eg,pm,pX,V = self.x
		return (self.Hcool+self.Hleak)/self.Area
	def Pcgf(self):						# fragmentation overpressure
		P,T,eg,pm,pX,V = self.x
		try: 
			_ = (xi for xi in T)
			Pcg = 1.e6/eg
			inds = np.where(eg<1.e-3)
			Pcg[inds] = 1.e9
		except TypeError:
			Pcg = 1.e6/np.max([eg,1.e-3])
		return Pcg

		
	##################
	##################    DERIVATIVE EQUATIONS
	##################	
	def dPdtf(self):					# pressure derivative wrt time
		P,T,eg,pm,pX,V = self.x 
		return -((self.Bmw*self.Ch - self.Cmw*self.Bh)*self.Dm+(self.Cm*self.Bh - self.Bm*self.Ch)*self.Dmw+(self.Bm*self.Cmw - self.Cm*self.Bmw)*self.Dh)/self.det
	def dTdtf(self):					# temperature derivative wrt time
		P,T,eg,pm,pX,V = self.x
		return -((self.Cmw*self.Ah - self.Amw*self.Ch)*self.Dm+(self.Am*self.Ch - self.Cm*self.Ah)*self.Dmw+(self.Cm*self.Amw - self.Am*self.Cmw)*self.Dh)/self.det
	def degdtf(self):					# gas fraction derivative wrt time
		P,T,eg,pm,pX,V = self.x
		degdt = -((self.Amw*self.Bh - self.Bmw*self.Ah)*self.Dm+(self.Bm*self.Ah - self.Am*self.Bh)*self.Dmw+(self.Am*self.Bmw - self.Bm*self.Amw)*self.Dh)/self.det
		return degdt
	def dpmdtf(self):					# melt density derivative wrt time
		P,T,eg,pm,pX,V = self.x
		return self.dpmdP*self.dPdt+self.dpmdT*self.dTdt
	def dpXdtf(self):					# crystal density derivative wrt time
		P,T,eg,pm,pX,V = self.x
		return self.dpXdP*self.dPdt+self.dpXdT*self.dTdt
	def dVdtf(self):					# volume derivative wrt time
		P,T,eg,pm,pX,V = self.x
		return (self.dPdt/self.br+self.deltaP/self.etar-self.ar*self.dTdt)*V

		
	##################
	##################    RUN MODEL
	##################
	def intermediate_vars(self):		# call thermodynamic functions above
		# gas density and derivatives
		self.pg = self.pgf()
		self.dpgdP = self.dpgdPf()
		self.dpgdT = self.dpgdTf()
		
		# crystal fraction and derivatives 
		self.eX = self.eXf()
		self.deXdT = self.deXdTf()
		self.deXdeg = self.deXdegf()
		
		# solubility and derivatives
		self.meq = self.meqf()
		self.dmeqdP = self.dmeqdPf()
		self.dmeqdT = self.dmeqdTf()
		
		# density component derivatives
		self.dpXdP = self.dpXdPf()
		self.dpXdT = self.dpXdTf()
		self.dpmdP = self.dpmdPf()
		self.dpmdT = self.dpmdTf()
		
		# density and derivatives
		self.em = self.emf()
		self.p = self.pf()
		self.dpdP = self.dpdPf()
		self.dpdT = self.dpdTf()
		self.dpdeX = self.dpdeXf()
		self.dpdeg = self.dpdegf()
		
		# effective properties
		self.b_1 = self.b_1f()
		self.a = self.af()
		
		# specific heat and derivatives
			# product of specific heat and density
		self.cxp = self.cxpf()
		self.c = self.cf()
		self.dcdP = self.dcdPf()
		self.dcdT = self.dcdTf()
		self.dcdeg = self.dcdegf()
		
		# overpressure
		self.deltaP = self.deltaPf()
		self.Pcg = self.Pcgf()
		
		# sources and sinks
		self.zbdt = self.zbdtf()			# compute BDT
		self.dzs = self.dzsf()				# compute shell thickness
		self.Pbdt = self.Pbdtf()			# hydrostatic pressure at BDT
		self.bg = self.bgf()				# compressibility relation
			# enthalpy out
		self.temperature_field()
			# water in
				# crystal fraction incoming magma
		self.eXin = self.eXinf()
				# melt mass fraction incoming magma (no exsolved gas)
		self.Mmin = self.Mminf()
				# water fraction dissolved in incoming melt
		self.Mwin = self.Mwinf()
			# water out
		self.Mwout = self.Mwoutf()
			# enthalpy in
		self.cin = self.cinf()
		self.Hin = self.Hinf()
		self.Hout = self.Houtf()
	def temperature_field(self):		# calculate temperature in shell
		# TBD
		self.compute_cooling()
		#self.compute_viscosity()
		self.etar = self.etarf()
	def compute_cooling(self):			# calculate cooling through shell
		# 1D sill model - equilibrium
		if self.geom == 1:
			self.Area = self.Areaf()
			self.Hcool = self.Hcoolf()
		# 3D D&H2014 model - disequilibrium
		elif self.geom == 3:
			self.R0 = (self.Vi*0.75/np.pi)**(1./3)
			self.S = 11*self.R0
			if self.t > self.i_s*self.dt_s:				
				self.update_cooling()
				
			tol = 1.e-4
			F = 0.
			n = 1
			dF = self.dFnt(n)
			F += dF
			n += 1
			err=10*tol
			while err > tol:
				dF = self.dFnt(n)
				err = abs(dF)/abs(1+(F+dF))
				print(dF,F,err)
				F += dF
				n += 1
				if n > 50:
					raise SimulationError('check dTdr calculation')
					
			self.Hcool = -self.k*4*np.pi*self.R0*F/(self.S-self.R0)
		else:
			raise SimulationError('unrecognized geometry')
	def dFnt(self,n):					# NOT WORKING
		b = n*np.pi/(self.S-self.R0)
		e = np.exp(-self.kappa*b**2*self.t)
		dFn = 2*e/(self.S-self.R0)*(self.R0*self.S*(self.Ti-self.TS)*(1-np.cos(n*np.pi))+(self.S*self.TS-self.R0*self.Ti)*(self.R0-self.S*np.cos(n*np.pi)))
		if self.i_s > 0:
			asdf
			ej = np.exp(-self.kappa*b**2*(self.t-self.t_s[:self.i_s]))
			ej1 = np.exp(-self.kappa*b**2*(self.t-self.t_s[1:self.i_s+1]))
			dFn += 2*self.R0*np.sum(self.m_s[:self.i_s]/(self.kappa*b**2)*()+self.p_s[:self.i_s]*(ej1+ej))
			dFn += -2*self.S*self.TS*np.cos(n*np.pi)*np.sum(ej1-ej)
		return dFn
	def update_cooling(self):			# NOT WORKING
		self.i_s += 1
			# linearly interpolate temperature back to previous step
		T0 = self.T_s[self.i_s-1]
		t0 = self.t_s[self.i_s-1]
		Ti = self.x[1]
		ti = self.t
		self.T_s[self.i_s] = T0 + (Ti-T0)/(ti-t0)*(t-t0)
		self.m_s[self.i_s] = np.diff(self.T_s[self.i_s-1:self.i_s+1])/self.dt_s
		self.p_s[self.i_s] = self.T_s[self.i_s]-self.m_s[self.i_s]*self.t_s[self.i_s]
	def etarf(self):					# calculate effective viscosity in shell
		P,T,eg,pm,pX,V = self.x
		if self._etari is not None:
			return self._etari+0.*P
		else:
			return self.A*np.exp(self.G/(self.B*T))
	def coefficients(self):				# apply cramer's rule
		# unpack state
		P,T,eg,pm,pX,V = self.x
		
		# mass balance (as in D&H2014)
		self.Am = -(self.b_1+1./self.br)
		self.Bm = -(-self.a-self.ar+self.deXdT*(pX-pm)/self.p)
		self.Cm = -(self.pg-pm + (pX-pm)*self.deXdeg)/self.p
		self.Dm = (self.Min(self.t)-self.Mout-self.Mwleak)/(V*self.p)-self.deltaP/self.etar

		# water balance (pg*eg*D&H2014 eqn)
		self.Amw = -(eg*self.dpgdP + eg*self.pg/self.br + pm*self.em*(self.dmeqdP +self.meq/self.bm + self.meq/self.br))
		self.Bmw = -(eg*(self.dpgdT - self.ar*self.pg)+pm*(self.em*self.dmeqdT-self.meq*self.em*self.am-self.meq*self.em*self.ar-self.meq*self.deXdT))
		self.Cmw = -(self.pg -self.meq*pm*(self.deXdeg + 1.))
		self.Dmw = (self.Mwin - self.Mwout)/V - self.deltaP*(eg*self.pg+self.meq*pm*self.em)/self.etar

		# energy balance (as in D&H2014)
		self.Ah = -(self.Lpv*P/(self.cxp*T*self.br) + self.b_1 + self.dcdP/self.c + 1./self.br - self.Lm*pX*self.eX/(self.cxp*T)*(1./self.bX + 1./self.br) - self.Le*pm*self.em/(self.cxp*T)*(self.dmeqdP + self.meq/self.bm + self.meq/self.br))
		self.Bh = -(-self.Lpv*self.ar*P/(self.cxp*T) - self.a - (pX-pm)/self.p*self.deXdT + self.dcdT/self.c + 1./T - self.ar - self.Lm*pX/(self.cxp*T)*(-self.aX*self.eX + self.deXdT - self.ar*self.eX) - self.Le*pm/(self.cxp*T)*(self.dmeqdT*self.em - self.meq*self.am*self.em - self.meq*self.deXdT - self.meq*self.ar*self.em))
		self.Ch = -((self.pg-pm)/self.p + (pX-pm)/self.p*self.deXdeg + self.dcdeg/self.c - self.Lm*pX/(self.cxp*T)*self.deXdeg + self.Le*self.meq*pm/(self.cxp*T)*(1.+self.deXdeg))
		self.Dh = (self.Hin - self.Hout)/(T*V*self.cxp) - self.deltaP/self.etar*(1. - self.Lm*pX*self.eX/(T*self.cxp) - self.Le*self.meq*pm*self.em/(T*self.cxp) - self.Lpv*P/(T*self.cxp))
		
		if self.dry:
			#self.det = (self.Am*self.Bh - self.Ah*self.Bm)
			zerocoefs = ['Cm','Amw','Bmw','Dmw','Ch']
			for coef in zerocoefs: 
				self.__setattr__(coef,0.)
			self.Cmw = -1.
			
		# determinant
		self.det = (self.Cm*(self.Amw*self.Bh - self.Bmw*self.Ah) + self.Bm*(self.Cmw*self.Ah - self.Amw*self.Ch) + self.Am*(self.Bmw*self.Ch - self.Cmw*self.Bh))

		
	##################
	##################    POST-PROCESSING
	##################	
	def Pf(self):						# pressure
		P,T,eg,pm,pX,V = self.x
		return P
	def Tf(self):						# temperature
		P,T,eg,pm,pX,V = self.x
		return T-273
	def egf(self):						# gas fraction
		P,T,eg,pm,pX,V = self.x
		return eg
	def pmf(self):						# melt density
		P,T,eg,pm,pX,V = self.x
		return pm
	def pXf(self):						# crystal density
		P,T,eg,pm,pX,V = self.x
		return pX
	def Vf(self):						# volume
		P,T,eg,pm,pX,V = self.x
		return V
	def Mf(self):						# total mass
		P,T,eg,pm,pX,V = self.x
		return self.p*V
	def Mwf(self):						# total water
		P,T,eg,pm,pX,V = self.x
		return self.pg*eg*V + self.meq*pm*self.em*V
	def Mwexsolf(self):					# exsolved water
		P,T,eg,pm,pX,V = self.x
		return self.pg*eg*V
	def Mwdissolf(self):				# dissolved water
		P,T,eg,pm,pX,V = self.x
		return self.meq*pm*self.em*V
	def Hf(self):						# total enthalpy
		P,T,eg,pm,pX,V = self.x
		return self.p*self.c*T*V - self.Lm*pX*self.eX*V - self.Le*self.meq*pm*self.em*V + P*(V-self.Vi)
	def Hsensf(self):					# sensible heat
		P,T,eg,pm,pX,V = self.x
		return self.p*self.c*T*V
	def Hcrysf(self):					# latent heat of crystallisation
		P,T,eg,pm,pX,V = self.x
		return -self.Lm*pX*self.eX*V
	def Hexsolf(self):					# latent heat of exsolution
		P,T,eg,pm,pX,V = self.x
		return -self.Le*self.meq*pm*self.em*V
	def Hpvf(self):						# pressure-volume work
		P,T,eg,pm,pX,V = self.x
		return P*(V-self.Vi)
	def Mbalf(self):					# cumulative mass balance
		return cumtrapz(self.dMbaldt, self.t, initial=0)
	def dMbaldtf(self):					# instantaneous mass balance
		return self.Am*self.dPdt+self.Bm*self.dTdt+self.Cm*self.degdt+self.Dm
	def Mwbalf(self):					# cumulative water balance
		return cumtrapz(self.dMwbaldt, self.t, initial=0)
	def dMwbaldtf(self):				# instantaneous water balance
		return self.Amw*self.dPdt+self.Bmw*self.dTdt+self.Cmw*self.degdt+self.Dmw
	def Hbalf(self):					# cumulative enthalpy balance
		return cumtrapz(self.dHbaldt, self.t, initial=0)
	def dHbaldtf(self):					# instantaneous enthalpy balance
		return self.Ah*self.dPdt+self.Bh*self.dTdt+self.Ch*self.degdt+self.Dh
	def Veruptf(self):					# cumulative erupted volume
		Verupt = 0.*self.t
		erupting = False
		i = 0
		for tii in self.te[1:]:
			while self.t[i] <= tii and i<len(self.t)-1:
				if erupting:
					Verupt[i] = Verupt[i-1] + self.Merupt/self.p[i]*(self.t[i]-self.t[i-1])
				else:
					Verupt[i] = Verupt[i-1]
				i += 1
				if i == len(self.t)-1:
					return Verupt
			erupting = not erupting	
	def transform(self, y, var):		# call function to transform model output
		# apply each scaling if appropriate
		for scale in scales:
			if var in scale[0]:
				y = y*scale[1]
				
		# vectorise
		vectorize = ['eXin','Mmin','Mwin','Pc']
		if var in vectorize:
			try:
				_ = (xi for xi in y)
			except TypeError:
				return y+0.*self.t
		
		# if not caught, return unchanged
		return y
	def set_eruption_rate(self):		# assigns a variable eruption rate
		self.Mout = 0.*self.t
		self.Verupt = 0.*self.t
		erupting = False
		i = 0
		for tii in self.te[1:]:
			while self.t[i] <= tii and i<len(self.t)-1:
				if erupting:
					self.Mout[i] = self.Merupt
				i += 1
				if i == len(self.t)-1:
					return
			erupting = not erupting		


			
##################
##################    OTHER OBJECTS
##################	
class JSONMC(object):					# output save file
	def __init__(self,**kwargs):
		for k in kwargs:
			try:
				self.__setattr__(k,kwargs[k].tolist())
			except AttributeError:
				self.__setattr__(k,kwargs[k])
class MinDist(object):					# non-constant recharge distribution
	def __init__(self, avgdt, dvavg, dvscale, dtdv, dvmax, dvshape=3, ndt=None, seed=None, constant=None, vector = None):
		self.avgdt = avgdt
		self.ndt = ndt
		self.dvavg = dvavg
		self.dvscale = dvscale
		self.dtdv = dtdv
		self.dvmax = dvmax
		self.dvshape = dvshape
		self.seed = seed
		self.constant = False
		self.vector = False
		self.Min_constant = None
		self.Min_vector = None
		if constant is not None:
			self.constant = True
			self.Min_constant=constant
		if vector is not None:
			self.vector = True
			assert len(vector) == 2
			self.tv = vector[0]
			self.Mv = vector[1]
	def __repr__(self):
		return "Min dist."
	def sample(self,tmax):
		# skip this if constant Min given
		if self.constant or self.vector:
			return
		# seed distribution if requested
		if self.seed is not None:
			np.random.seed(seed=self.seed)
		# compute ndt if not given, compute from tmax
		if self.ndt is None:
			self.ndt = int(tmax/self.avgdt)
		
		self.dts = expon.rvs(scale=self.avgdt*2.,size=self.ndt)
		self.dvs = lognorm.rvs(self.dvshape,loc=self.dvavg,scale=self.dvscale,size=self.ndt)
		self.dvs[np.where(self.dvs>self.dvmax)]=self.dvmax
		self.ts = np.cumsum(self.dts)
		
		dvdti = lambda t, dv, te, s: dv*np.exp(-0.5*((t-te)/s)**2)/(s*np.sqrt(2*np.pi))
		dvdt = lambda t: np.sum(map(partial(dvdti, s=self.dtdv), self.dvs, self.ts))
	def __call__(self,t):
		try:
			# if iterable, call function recursively
			return np.array([self.__call__(ti) for ti in t])
		except:
			pass 
			
		if self.constant:
			return self.Min_constant
		elif self.vector:
			return np.interp(t,self.tv,self.Mv)
		else:
			return np.sum([dvi*np.exp(-0.5*((t-tsi)/self.dtdv)**2)/(self.dtdv*np.sqrt(2*np.pi)) for tsi,dvi in zip(self.ts, self.dvs)])
class GammaDist(object):				# non-constant geothermal power				
	def __init__(self, base, freq, amp, phase=0., constant=None):
		self.base = base
		self.freq = freq
		self.amp = amp
		self.phase = phase
		self.constant = False
		self.gamma_constant = None
		if constant is not None:
			self.constant = True
			self.gamma_constant=constant
		else:
			if self.base < abs(self.amp) or self.base<0:
				raise SimulationError('Gamma distribution as defined will generate negative values')
	def __repr__(self):
		return "Gamma dist."
	def __call__(self,t):
		try:
			# if iterable, call function recursively
			return np.array([self.__call__(ti) for ti in t])
		except:
			pass 
			
		if self.constant:
			return self.gamma_constant
		else:
			return self.amp*np.sin(self.freq*t+self.phase)+self.base

			
##################
##################    SUPPORTING FUNCTIONS
##################	
def dens(P,T):							# equation of state for liquid water
	''' return fluid density as function of pressure and temperature (FEHM formulation)
	'''
	
	YL0 = (YDL[0]+YDL[1]*P+YDL[2]*P**2+YDL[3]*P**3+YDL[4]*T+YDL[5]*T**2+YDL[6]*T**3+YDL[7]*P*T+YDL[8]*P**2*T+YDL[9]*P*T**2)
	ZL0 = (ZDL[0]+ZDL[1]*P+ZDL[2]*P**2+ZDL[3]*P**3+ZDL[4]*T+ZDL[5]*T**2+ZDL[6]*T**3+ZDL[7]*P*T+ZDL[8]*P**2*T+ZDL[9]*P*T**2)
	return YL0/ZL0
def get_event_time(te):					# determine which ODE event triggered and when
	for tei in te:
		if len(tei)>0:
			return tei[0]
def dxdt(x,t,p,verbose=False):			# canonical vector derivative wrt time
	''' Derivative equation for ODEINT
		
		parameters
		----------
		x : array-like
			contains the current state of the 6 canonical variables
		t : float
			time
		p : par object
			object containing relevant parameters as attributes
			
		returns
		-------
		dxdt : array-like
			contains time derivatives of the 6 canonical variables
			
	'''
	# set state and time attributes
	p.x = x
	p.t = t
	if p.verbose:
		print('t={:4.3e}, P={:4.3e}, T={:4.3e}, eg={:4.3e}, pm={:4.3e}, pX={:4.3e}, pV={:4.3e}'.format(p.t,*p.x))
	# compute intermediate variables (e.g., total density)
	p.intermediate_vars()
	# compute coefficients for cramer's rule
	p.coefficients()
	# compute derivatives
		# pressure
	p.dPdt = p.dPdtf()
		# temperature
	p.dTdt = p.dTdtf()
		# gas fraction
	p.degdt = p.degdtf()
		# melt density
	p.dpmdt = p.dpmdtf()
		# crystal density
	p.dpXdt = p.dpXdtf()
		# volume
	p.dVdt = p.dVdtf()
	
	derivatives = [p.dPdt, p.dTdt, p.degdt, p.dpmdt, p.dpXdt, p.dVdt]
	if verbose:
		print(('{:6.5e}, '*(len(derivatives)+1))[:-2].format(t,*derivatives))
	return derivatives
def eruption_condition1(x,t,p):			# condition to trigger eruption - dike propagation
	P,T,eg,pm,pX,V = x
	if p.verbose:
		print('eruption 1: dP-dPc={:3.2f}MPa, eX={:3.2f}'.format((P-(p.Plith+p.Pc))/1.e6, p.eX))
	return np.min([(P-(p.Plith+p.Pc))/1.e8,0])+np.min([p.eXc-p.eX,0])
def eruption_condition2(x,t,p):			# condition to trigger eruption - fragmentation
	P,T,eg,pm,pX,V = x
	if p.verbose:
		print('eruption 2: dP-dPg={:3.2f}MPa, eX={:3.2f}'.format((P-(p.Plith+p.Pcg))/1.e6, p.eX))
	return np.min([(P-(p.Plith+p.Pcg))/1.e8,0])+np.min([p.eXc-p.eX,0])
def dormancy_condition1(x,t,p):			# condition to halt eruption - pressure drop to litho
	P,T,eg,pm,pX,V = x
	if p.verbose:
		print('dormancy 1: dP={:3.2f}MPa'.format((P-(p.Plith-p.uPc))/1.e6))
	return (P-(p.Plith-p.uPc))/1.e6
def dormancy_condition2(x,t,p):			# condition to halt eruption - crystallinity > 50%
	P,T,eg,pm,pX,V = x
	eX =(1-eg)*(1-((T-p.Ts)/(p.Tl-p.Ts))**p.b)
	if p.verbose:
		print('dormancy 2: eX-{:3.2f}={:3.2f}'.format(p.eXc,eX-p.eXc))
	return eX-p.eXc
def plot_model(model,var1,var2,var3,var4,var5,var6,plot_options,save=None):
	f = plt.figure(figsize=(14,4.5))
	ax1 = plt.axes([0.06,0.13,0.22,0.6])
	ax3 = plt.axes([0.39,0.13,0.22,0.6])
	ax5 = plt.axes([0.72,0.13,0.22,0.6])
	ax2 = ax1.twinx()
	ax4 = ax3.twinx()
	ax6 = ax5.twinx()
	# axis for legend
	ax7 = plt.axes([0.04, 0.77, 0.90, 0.2])
	ax7.set_xticks([])
	ax7.set_yticks([])
	ax7.set_xlim([0,1])
	ax7.set_ylim([0,1])
	ax7.axis('off')
	
	axs = [ax1,ax2,ax3,ax4,ax5,ax6]
	cols = ['k','b','r','g','m','c']
	
	vars = [var1,var2,var3,var4,var5,var6]
		
	for modeli,ls in zip(model, ['-','--',':']):
		t,ys = modeli.get(vars,plot_options)
		for j,ax,var,y,col in zip(range(6),axs,vars,ys,cols):
			if var == '-':
				continue
			lbl = labels[var]
			ax.plot(t,y,col+ls)
			jx = j//2
			jy = j%2
			dx,dy = [0.365,0.3]
			x0 = 0.03
			ax7.plot([jx*dx+x0,jx*dx+x0+0.03],[0.8-jy*dy,0.8-jy*dy],'-',color=col)
			ax7.text(jx*dx+x0+0.04,0.8-jy*dy,lbl,color=col, ha='left',va='center')
			ax.tick_params(axis='y',colors=col)
			
		tlabel = r'$t$, time [yrs]'
	
	# edge cases
		# lock axes when comparing pressures
	ps = ['deltaP','Pc','Pcg']
	for i in range(3):
		if vars[2*i] in ps and vars[2*i+1] in ps:
			ylim1 = axs[2*i].get_ylim()
			ylim2 = axs[2*i+1].get_ylim()
			for ax in [axs[2*i], axs[2*i+1]]: 
				ax.set_ylim([np.min([ylim1[0],ylim2[0]]),np.max([ylim1[1],ylim2[1]])])
	
	col = [0.5,0.5,0.5]
	for jx,ls in zip(range(len(model)),['-','--',':']):
		jy = 2
		ax7.plot([jx*dx+x0,jx*dx+x0+0.03],[0.8-jy*dy,0.8-jy*dy],ls,color=col)
		mstr = 'model {:d}'.format(jx+1)
		if model[jx].label != '':
			mstr += ' - '+model[jx].label
		ax7.text(jx*dx+x0+0.04,0.8-jy*dy,mstr,color=col, ha='left',va='center')
	
	ax1.set_xlabel(tlabel)
	ax3.set_xlabel(tlabel)
	ax5.set_xlabel(tlabel);
	if save is None:
		plt.show()
	else:
		plt.savefig(save,dpi=400)
		plt.clf()
def save_plot(controls,save):			# save notebook plot to disk
	var1 = controls['var1'].value
	var2 = controls['var2'].value
	var3 = controls['var3'].value
	var4 = controls['var4'].value
	var5 = controls['var5'].value
	var6 = controls['var6'].value
	plo = controls['plot_options'].value
	model = controls['model'].value
	plot_model(model,var1,var2,var3,var4,var5,var6,plo,save)
def plot_parts(models, show = ['deltaP','Pcg','eX','T','eg','Wgeo']):					# parts for interactive Jupyter widget
	# process models as list or not
	try:
		_ = (xi for xi in models)
	except TypeError:
		models = [models,]
	
	ts = ''
	if len(models) == 1:
		ts += models[0].terminate_string
	else:
		for i, model in enumerate(models):
			ts += '({:d}) '.format(i+1)+model.terminate_string+' <br />'
		ts = ts[:-2]
	
	# QA defaults
	while len(show) < 6:
		show.append('-')
	for i in range(6):
		if show[i] not in funcs:
			show[i] = '-'
	
	ts2 = summary_text(models)
	
	erp = RadioButtons(options=['full simulation','latest dormancy','latest eruption'], value='full simulation',description='Show:')
	var1 = Dropdown(options=dropdown_options,value=show[0],description='')
	var2 = Dropdown(options=dropdown_options,value=show[1],description='')
	var3 = Dropdown(options=dropdown_options,value=show[2],description='')
	var4 = Dropdown(options=dropdown_options,value=show[3],description='')
	var5 = Dropdown(options=dropdown_options,value=show[4],description='')
	var6 = Dropdown(options=dropdown_options,value=show[5],description='')
	txt = HTMLMath(value=ts,description='status:')
	txt2 = HTMLMath(value=ts2)
	savetxt = Text(value='model_run.png')
	savebutton = Button(description='SAVE', tooltip='save a copy of the current figure')
	savefunc = lambda x: save_plot(controls,savetxt.value)
	savebutton.on_click(savefunc)
	
	box = [txt2,
		   HBox([erp, VBox([txt, HBox([savetxt, savebutton])])]),
			HBox([VBox([var1,var2]),VBox([var3,var4]),VBox([var5,var6])])]
					  
	controls = {'var1':var1,'var2':var2,'var3':var3,'var4':var4,'var5':var5,'var6':var6,'plot_options':erp}
	controls.update({'model':fixed(models)})
	return box, controls
def summary_text(models):				# HTML text for summary table
	post_vars = [
	['Ne','Eruptions','{:d}'],
	['dtavg','Avg. eruption period</br>[yr]','{:4.0f}'],
	['dVavg','Avg. erupted volume</br>[km<sup>3</sup>]','{:4.3f}'],
	['egWavg','Avg. volatiles in geothermal','{:3.2f}'],
	['dwdtavg','Avg. uplift rate</br>[mm/yr]','{:3.2f}'],
	['Wavg','Avg.&nbspgeothermal flux</br>[W m<sup>-2</sup>]','{:3.2f}'],
	['eXavg','Avg. erupted crystal fraction</br>','{:3.2f}'],
	['egavg','Avg. erupted volatile fraction</br>','{:3.2f}'],
	['dTdtavg','Avg.&nbspcooling rate</br>[<sup>o</sup>C kyr<sup>-1</sup>]','{:4.2f}'],
	]
	
	
	ts = ''
	ts += '<tr>'
	ts += '<th></th>'
	for post_var in post_vars:
		ts += '<th>{:s}</th>'.format(post_var[1])
	ts += '</tr>'
	
	for i,model in enumerate(models):
		ts += '<tr>'
		ts += '<th>model&nbsp{:d}</th>'.format(i+1)
		for post_var in post_vars:
			if model.__getattribute__(post_var[0]) is None:
				ts += '<td>-</td>'
			else:
				y = model.__getattribute__(post_var[0])
				y = model.transform(y,post_var[0])
				ts += ('<td>'+post_var[2]+'</td>').format(y)
		ts += '</tr>'
	
	ts2 = ''	
	with open('table_template.html','r') as fp:
		for ln in fp.readlines():
			ts2 += ln.rstrip()
	
	st = '<table style="width:100%">'
	i1 = ts2.find(st)+len(st)
	i2 = ts2.find('</table>')
	return ts2[:i1] + ts + ts2[i2:]
def show_model(models,show):			# notebook commands to create widget
	box,controls = plot_parts(models, show)
	return VBox([*box, interactive_output(plot_model, controls)])