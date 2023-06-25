#Exact riemann solver for 1D eular
#The code consist of 3 parts- one that executes the solver and saves the output to an output file, one contains the defined functions and other is a simple plotting program
#For use, split this python file to the following parts for convenience.

#######################################################################################################################################################
#part 1: solver.py (executes the solver)

#execute solver.py from the terminal.
#As user input it takes the left state, right state, number of points, time at which the solution is needed, ideal gas ratio gamma and the bound of the spatial grid.
#If no inputs are given, then the solver takes the default defined values, which is specified for shock-tube problem
#######################################################################################################################################################
import argparse
import riemann
import sys
import ast

def setupParser():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-l_state', '--stateL', dest = 'stateL', default='[1,0.75,1.0]', help='set user left state [dens, vel, pres]')
	parser.add_argument('-r_state', '--stateR', dest = 'stateR', default='[0.125,0.0,0.1]', help='set user right state [dens, vel, pres]')
	parser.add_argument('-x0', '--x0', dest = 'x0', default = '0.5', help = 'set X0')
	parser.add_argument('-t', '--time', dest = 'time', default = '0.2', help = 'set time')
	parser.add_argument('-g','--gamma',dest='gamma', default=1.4, help='set ideal gas gamma')	
	parser.add_argument('-n','--npts',dest='npts', default=1000, help='set number of evaluation points')
	parser.add_argument('-b','--bounds',dest='bounds', default='[0.0,1.0]', help='set domain boundaries')
	return parser

if(__name__ == '__main__'):
	parser = setupParser()
	args = parser.parse_args()

	# parse parameters
	user_def = 'sod_problem'
	gamma = float(args.gamma)
	npts  = int(args.npts)
	xbnds = ast.literal_eval(args.bounds)
	stateL = ast.literal_eval(args.stateL)
	stateR = ast.literal_eval(args.stateR)
	t      = ast.literal_eval(args.time)
	x0     = ast.literal_eval(args.x0)
	
        # terminal
	sys.stdout.write('Running problem: {:s}\n'.format(user_def))
	sys.stdout.write('  xbnds: {:f} {:f}\n'.format(xbnds[0],xbnds[1]))
	sys.stdout.write('  x0: {:f}\n'.format(x0))
	sys.stdout.write('  grid:  number of points {:n}\n'.format(npts))
	sys.stdout.write('  left:  {:f}  {:f}  {:f}\n'.format(stateL[0], stateL[1], stateL[2]))
	sys.stdout.write('  right: {:f}  {:f}  {:f}\n'.format(stateR[0], stateR[1], stateR[2]))
	sys.stdout.write('  gamma: {:f}\n'.format(gamma))
	sys.stdout.write('  time:  {:f}\n'.format(t))
	sys.stdout.write('  x0:  {:f}\n'.format(x0))

	rp = riemann.riemann(gamma, stateL, stateR)
	success = rp.solve()
	if(not success):
		sys.stdout.write('[FAILURE] Unable to solve problem {:s}'.format(user_def))
	else: pass

	x  = []
	s  = []
	shift = 0.0
	dx = (xbnds[1]-xbnds[0])/npts
	shift = 0.5e0
	for i in range(0,npts):
		x.append(xbnds[0] + (float(i)+shift)*dx)
		if(abs(t) <= 1.0e-9):
			s.append(x[i]-x0)
		else:
			s.append((x[i]-x0)/t)
	dens, pres, velx, eint, cspd = rp.sample(s)

	# write output to a .dat file
	filename = 'output/exact_rp_{:s}.dat'.format(user_def)
	sys.stdout.write('  Writing to file: {:s}\n'.format(filename))
	with open(filename,'w+') as file:
		file.write('# Exact Riemann solution for problem: {:s}\n'.format(user_def))
		file.write('# Domain bounds: [{:<.5e},{:.5e}]\n'.format(xbnds[0],xbnds[1]))
		file.write('# Iterface position: {:.5e}\n'.format(x0))
		file.write('# Gamma: {:.5e}\n'.format(gamma))
		file.write('# Left  state [dens,velx,pres]: [{:<.5e}, {:.5e}, {:.5e}]\n'.format(stateL[0],stateL[1],stateL[2]))
		file.write('# Right state [dens,velx,pres]: [{:<.5e}, {:.5e}, {:.5e}]\n'.format(stateR[0],stateR[1],stateR[2]))
		file.write('# Time: {:.5e}\n'.format(t))
		file.write('{:>17} {:>17} {:>17} {:>17} {:>17} {:>17}\n'.format('x','dens','pres','velx','eint','cspd'))
		for i in range(0,npts):
			file.write('{:17.9e} {:17.9e} {:17.9e} {:17.9e} {:17.9e} {:17.9e}\n'.format(x[i],dens[i],pres[i],velx[i],eint[i],cspd[i]))
			
#######################################################################################################################################################
#part2: riemann.py (containes all the defined functions)
#######################################################################################################################################################
from math import sqrt,fabs,pow

class init_state:
	def __init__(self, ideal_gas_gamma, left_state, right_state):
		self.gamma = ideal_gas_gamma
		self.left_den = float(left_state[0])
		self.left_vel = float(left_state[1])
		self.left_press = float(left_state[2])
		self.right_den = float(right_state[0])
		self.right_vel = float(right_state[1])
		self.right_press = float(right_state[2])
		
class riemann(init_state):
	def __init__(self, ideal_gas_gamma, left_state, right_state):	
		super(riemann, self).__init__(ideal_gas_gamma, left_state, right_state)
		self.L_cs = sqrt(self.gamma*self.left_press/self.left_den)
		self.R_cs = sqrt(self.gamma*self.right_press/self.right_den)		
		self.G1 = 0.5e0*(self.gamma - 1.e0)/self.gamma
		self.G2 = 0.5e0*(self.gamma + 1.e0)/self.gamma
		self.G3 = 1.e0/self.G1
		self.G4 = 1.e0/(self.G1*self.gamma)
		self.G5 = 1.e0/(self.G2*self.gamma)
		self.G6 = self.G1/self.G2
		self.G7 = self.G1*self.gamma
		self.G8 = self.gamma - 1.e0
		
		self.star_press = 0.e0
		self.star_vel = 0.e0
		self.success = False
		
	def solve(self):
		self.success = True
		if (self.G4*(self.L_cs+self.R_cs)<=(self.right_vel-self.left_vel)):
			print('Initial state will generate vacuum: Terminating solver')
			self.success = False
			return False
		success = self.star_press_vel()
		if(not success):
			print('unable to calculate pressure and velocity in the star region')
			self.success = False
			return False
		return True
		
	def pressure_function(self, P, density_K, press_K, K_cs):
		F  = -1e99
		dF = -1e99
		if(P <= press_K):
			prat = P/press_K
			F    = self.G4*K_cs*(pow(prat, self.G1) - 1.e0)
			dF   = (1.e0/(density_K*K_cs))*pow(prat, -self.G2)
		else:
			aK  = self.G5/density_K
			bK  = self.G6*press_K
			qrt = sqrt(aK/(bK + P))
			F   = (P - press_K)*qrt
			dF  = (1.e0 - 0.5e0*(P - press_K)/(bK + P))*qrt
		return F, dF	
		
	def pressure_guesser(self):
		pm = -1e99
		q_user = 2.e0 #pressure ratio

		cup  = 0.25e0*(self.left_den + self.right_den)*(self.L_cs + self.R_cs)
		ppv  = 0.5e0*(self.left_press + self.right_press) + 0.5e0*(self.left_vel - self.right_vel)*cup
		ppv  = max([0.e0, ppv])
		pmin = min(self.left_press, self.right_press)
		pmax = max(self.left_press, self.right_press)
		q_max = pmax/pmin
		if( (q_max <= q_user) and (pmin <= ppv) and (ppv <= pmax) ):
			pm = ppv
		else:
			if(ppv < pmin):
				pq = pow(self.left_press/self.right_press, self.G1)
				um = (pq*self.left_vel/self.L_cs + self.right_vel/self.R_cs + self.G4*(pq - 1.e0))/(pq/self.L_cs + 1.e0/self.R_cs)
				ptL = 1.e0 + self.G7*(self.left_vel - um)/self.L_cs
				ptR = 1.e0 + self.G7*(um - self.right_vel)/self.R_cs
				pm  = 0.5e0*(self.left_press*pow(ptL,self.G3) + self.right_press*pow(ptR,self.G3))
			else:
				geL = sqrt((self.G5/self.left_den)/(self.G6*self.left_press + ppv))
				geR = sqrt((self.G5/self.right_den)/(self.G6*self.right_press + ppv))
				pm  = (geL*self.left_press + geR*self.right_press - self.right_vel + self.left_vel)/(geL + geR)
		return pm
			
	def star_press_vel(self):
		max_iteration = 1000
		TOL = 1e-6
		p_star = -1e99
		u_star = -1e99
		p_star_init = self.pressure_guesser()
		p_old   = p_star_init
		p_new   = p_old
		delta_u = self.right_vel - self.left_vel
		newton_raphson_success = False
		for it in range(1,max_iteration+1):
			f_L, f_prime_L = self.pressure_function(p_old, self.left_den, self.left_press, self.L_cs)
			f_R, f_prime_R = self.pressure_function(p_old, self.right_den, self.right_press, self.R_cs)
			p_new = p_old - (f_L + f_R + delta_u)/(f_prime_L + f_prime_R)
			dp = 2.e0*fabs((p_new - p_old)/(p_new + p_old))
			if(dp <= TOL):
				newton_raphson_success = True
				break
			if(p_new < 0.e0):
				p_new = TOL
			p_old = p_new
		if(not newton_raphson_success):
			print('unable to converge newton raphson scheme')
		else:
			p_star = p_new
			u_star = 0.5e0*(self.left_vel + self.right_vel + f_R - f_L)
		self.star_press = p_star
		self.star_vel = u_star
		return newton_raphson_success
		
	def sampler(self, S):
		d_Out = -1e99
		u_Out = -1e99
		p_Out = -1e99

		# Left of contact
		if(S <= self.star_vel):
			if(self.star_press <= self.left_press):
				shl = self.left_vel - self.L_cs
				if(S <= shl):
					d_Out = self.left_den
					p_Out = self.left_press
					u_Out = self.left_vel
				else:
					cml = self.L_cs*pow(self.star_press/self.left_press,self.G1)
					stl = self.star_vel - cml
					if(S > stl):
						d_Out = self.left_den*pow(self.star_press/self.left_press,1.0/self.gamma)
						u_Out = self.star_vel
						p_Out = self.star_press
					else:
						u_Out = self.G5*(self.L_cs + self.G7*self.left_vel + S)
						c    = self.G5*(self.L_cs + self.G7*(self.left_vel - S))
						d_Out = self.left_den*pow(c/self.L_cs,self.G4)
						p_Out = self.left_press*pow(c/self.L_cs,self.G3)
			else:
				pml = self.star_press/self.left_press
				sl  = self.left_vel - self.L_cs*sqrt(self.G2*pml + self.G1)
				if(S <= sl):
					d_Out = self.left_den
					p_Out = self.left_press
					u_Out = self.left_vel
				else:
					d_Out = self.left_den*(pml + self.G6)/(self.G6*pml + 1.e0)
					p_Out = self.star_press
					u_Out = self.star_vel
		# Right of contact
		else:
			if(self.star_press > self.right_press):
				pmr = self.star_press/self.right_press
				sr  = self.right_vel + self.R_cs*sqrt(self.G2*pmr + self.G1)
				if(S >= sr):
					d_Out = self.right_den
					p_Out = self.right_press
					u_Out = self.right_vel
				else:
					d_Out = self.right_den*(pmr + self.G6)/(self.G6*pmr + 1.e0)
					p_Out = self.star_press
					u_Out = self.star_vel
			else:
				shr = self.right_vel + self.R_cs
				if(S >= shr):
					d_Out = self.right_den
					p_Out = self.right_press
					u_Out = self.right_vel
				else:
					cmr = self.cspdR*pow(self.star_press/self.right_press,self.G1)
					stR = self.star_vel + cmr
					if(S <= stR):
						d_Out = self.right_den*pow(self.star_press/self.right_press,1.e0/self.gamma)
						p_Out = self.star_press
						u_Out = self.star_vel
					else:
						u_Out = self.G5*(-self.R_cs + self.G7*self.right_vel + S)
						c    = self.G5*(self.R_cs - self.G7*(self.right_vel - S))
						d_Out = self.right_den*pow(c/self.R_cs,self.G4)
						p_Out = self.right_press*pow(c/self.R_cs,self.G3)
		return d_Out, p_Out, u_Out
	
	def sample(self, s_points):
		den = []
		press = []
		vel = []
		eint = []
		cs = []

		for S in s_points:
			d, p, u = self.sampler(float(S))
			den.append(d)
			press.append(p)
			vel.append(u)
			eint.append((p/d)/(self.gamma-1.e0))
			cs.append(sqrt(self.gamma*p/d))

		return den, press, vel, eint, cs

#######################################################################################################################################################
#part 3: plotter.py (plotting tool)
#change the path to your local path where the python files are stored
#######################################################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
path = '/home/computational_methods/output/exact_rp_sod_problem.dat'
data = pd.read_csv(path, skiprows=7, delim_whitespace = True)

fig,axes = plt.subplots(nrows=3, ncols=1)
plt.subplot(3, 1, 1)
plt.plot(data['x'], data['dens'], 'k-')
plt.ylabel(r'$rho$',fontsize=16)
plt.tick_params(axis='x',bottom=False,labelbottom=False)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(data['x'], data['pres'], 'k-')
plt.ylabel('$p$',fontsize=16)
plt.tick_params(axis='x',bottom=False,labelbottom=False)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(data['x'], data['velx'], 'k-')
plt.ylabel('$u$',fontsize=16)
plt.tick_params(axis='x',bottom=False,labelbottom=False)
plt.grid(True)

plt.xlabel('x',fontsize=16)
plt.subplots_adjust(left=0.2)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(top=0.95)
plt.show()
fig.savefig("exact_solution_riemann.png", dpi=300)
