#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.interpolate import CubicSpline
from mesa.space import MultiGrid
from mesa import Agent, Model
from mesa.time import RandomActivation
import random
import matplotlib.pyplot as plt
from mesa.datacollection import DataCollector
from collections import Counter
from mesa.batchrunner import BatchRunner
import pandas as pd
from sympy import *
from sympy.stats import *
from sympy.solvers import solve
import pickle
from multiprocessing.dummy import Pool as ThreadPool
import time
from tqdm import tqdm
from scipy.stats import nbinom,binom,geom
import os


# In[ ]:


#Change this variable to exclude MSM from donating (True = Exclusion)
ExcludeMSM = True


# In[ ]:


#Probability Setup
agentnum=1

#Expiration time (in days) after donation by blood component
RBCDecay = 42
PlateletDecay = 6

#Days required between whole blood and platelet apheresis donations
WBtime =  56
Ptime =   7
RBCtime = 112

#Probability of having Type O+,A+,B+,O-,A-,AB+,B-,AB- respectively
BloodType = [.374,.357,.085,.066,.063,.034,.015,.006]

#Probability of having Breast,Prostate,Lung,Colon,Uterus,Melanoma skin,Urinary,Kidney,and Thyroid cancer
Cancer = [.000624,.000495,.000575,.00038,.000134,.000221,.000195,.000166,.000145]
CancerAndAnemiaAndTransfusionProbability = sum(Cancer)*.5 *.15
TimebetweenTransfusions = 105
RBCUnitsUsed = 2

#Probability of having a blood cancer and requiring Platelet transfusion
BloodCancerProbability = .00122 * .03
TotalPlateletsNeeded = 6

#HCT Prevalence
AllogeneicHCTchance = 7974/320000000/365
AutologousHCTchance = 12340/320000000/365
def Allounitsused():
	return(random.gauss(1,.2))
Autounitsused = 19

#Proportion of eligible people who actually donate on a given day (Create a probability distribution)
def WBDonate(day):
	return(1/2944.15*(-.000000000137718*day**5+.000000132706705*day**4-
								.000044760327769*day**3+.006167137934*day**2-.2983*day+10.393832))
density={}
for i in range(1,366):
	density[i] = WBDonate(i)

x = Symbol('x')
X = FiniteRV('X', density)

#Severe accident rate involving blood
GeneralTraumaRate =  .000416/365
PlateletTraumaRate = 151/479
PlasmaTraumaRate =   301/479
PlasmaperTrauma =    round(5163/301)
PlateletsperTrauma = round(1047/151)
k=1.2164
theta=9.6947
XX = Gamma('x', k, 1/theta)

#Sicklecell Prevalence
SCprev =  0.000302 

#Probability of MSM
PMSM = random.uniform(.02,.05)
MSMandHIV = 0.001917
NotMSMandHIV = 0.001509
MSMacquire =  26200/(321039839*PMSM*(1-MSMandHIV/PMSM))
NotMSMacquire = 8800/(321039839*(1-PMSM))

#Probability of HIV+ (MSM and not MSM, respectively)
PHIV=[MSMandHIV/PMSM,NotMSMandHIV/(1-PMSM)]

#Probability of being eligible to donate (given not MSM)
Eligibility= .38 + PMSM*(1-PHIV[0])

#Probability of acquiring HIV on a given day (MSM and not MSM)
AcHIV=[MSMacquire/365,NotMSMacquire/365]

#Probability of requiring RBC transfusion and average units used
PRBC=603000/316200000/365
NeededRBC=2.7 #units

#Probability of not knowing one has HIV
Dontknow=.15

#Net Daily population increase
PopIncrease=0.007/365

#Probability of death among those who know they have HIV
Death=6465/(.85*1122900)/365

#Cardiac Surgery Rate
Cardiac = 0.0011857/365
#Chance of RBC/platelet use in Cardiac Surgury
CardRBC = .27
CardPlat = .21
#Mean Units Used
CardRBCUnits = 5
CardPlatUnits = 2.2

#ICU admission rate
ICU = 0.013494133/365
#Chance of less than 8 day stay
stay=.948
#RBC Transfusion chance (less than 8 days and more than 8 days)
ICURBC=[.5,.8]
ICURBCUnits = 1

#GastroIntestinal Bleeding (RBC Use)
UpperGIBleedRate = .00067/365
MeanUpperGI = 2
LowerGIBleedRate = random.gauss(.00024,.0002)/365
MeanLowerGI = 2

#Hip replacement surgeries
Hip =  0.000968 


# In[ ]:


#DataCollector Functions                    

#Total HIV Cases
def computeHIV(model):
	agent_h = [a.HIV for a in model.schedule.agents]
	return(sum(agent_h))

#HIV Cases per 100,000
def computeHIVPrevalence(model):
	agent_h = [a.HIV for a in model.schedule.agents]
	x = sum(agent_h)
	N = model.num_agents
	return(x/N*100000)

#Amount of Blood in the Blood Bank
def BloodDonations(model):
	agent_Blood = [a.RBCBank for a in model.schedule.agents]
	return(sum(agent_Blood))

def PlateletDonations(model):
	agent_Blood = [a.PlateletBank for a in model.schedule.agents]
	return(sum(agent_Blood))

#Amount of Blood in the bank by Blood Type
def BloodbyType(model):
	TypeofBlood = {'O-':0,'O+':0,'A+':0,'A-':0,'B+':0,'B-':0,'AB+':0,'AB-':0}
	for a in model.schedule.agents:
		TypeofBlood[a.Blood] += a.RBCBank
	return(TypeofBlood)

def PlasmabyType(model):
	TypeofBlood = {'O-':0,'O+':0,'A+':0,'A-':0,'B+':0,'B-':0,'AB+':0,'AB-':0}
	for a in model.schedule.agents:
		TypeofBlood[a.Blood] += a.PlasmaBank
	return(TypeofBlood)

#Expired Units of Blood
def OutdatedStocks(model):
	return([model.outdatedRBC,model.outdatedPlatelet])

#Total Blood Donated up Until that Day
def TotalDonations(model):
	return([int(357000/315000000*model.num_agents)+sum([a.totRBC for a in model.schedule.agents]),
			int(70200/315000000*model.num_agents)+sum([a.totPlat for a in model.schedule.agents])])

#Total number of Failed HIV Donors (couldn't donate), Successful HIV donors, 
#and number of cases of HIV transfused into another person
def HIVDon(model):
	return([sum([a.HIVDetected for a in model.schedule.agents]),sum([a.HIVDonation for a in model.schedule.agents]),
			sum([a.ReceivedHIV for a in model.schedule.agents])])

#Distribution of Days since infection for Agents that Can Donate
def InfectionDays(model):
	List = []
	for a in model.schedule.agents:
		if a.Elig and a.HIV:
			List.append(a.Dayssinceinfection)
	return(List)

#Distribution of Days since infection for successful donors
def HIVDonors(model):
	List = []
	for a in model.schedule.agents:
		if a.HIVDonation:
			List.append(a.Dayssinceinfection)
	return(List)

#Distribution of Days since infection for donors
def FailedHIVDonors(model):
	List = []
	for a in model.schedule.agents:
		if a.HIVDetected:
			List.append(a.Dayssinceinfection)
	return(List)

def test(model):
	List = {}
	for a in model.schedule.agents:
		if a.HIVDonation:
			List[a.ID]=[a.Blood,a.RBCBank,a.PlateletBank,a.PlasmaBank]
		#elif a in list(List.keys()) and a.RBCBank <= 0:
		#    List.remove(a)
		#elif a in list(List.keys()) and a.PlateletBank <= 0:
		#    List1.remove(a)
	return(List)

def TotalHivBloodProducts(model):
	List = {}
	for a in model.schedule.agents:
		if a.HIVDonation:
			List[a.ID]=[a.Blood,a.RBCHIV,a.PlateletsHIV,a.PlasmaHIV]
		#elif a in list(List.keys()) and a.RBCBank <= 0:
		#    List.remove(a)
		#elif a in list(List.keys()) and a.PlateletBank <= 0:
		#    List1.remove(a)
	return(List)


# In[ ]:


#Probability of False Negative as a function of days since infection
Specificity = random.uniform(.998,.9999)
d=[0,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,34,40,42,45,50]
FN=[1,1,1,1,1,1,1,1,1,1,.99,.86,.79,.51,.4,.35,.31,.2,.18,.08,.08,.07,.05,.05,.01,.01,.005]
def FalseNegative(days):
	if days >= 50:
		return(1-Specificity)
	else:
		f=CubicSpline(d,FN)
		return(float(f(days)))
	
#Cumulative probability of acquiring HIV with last x number of days
def MassIncidence(MSM):
	if MSM:
		return(1-(1-AcHIV[0])**50)
	else:
		return(1-(1-AcHIV[1])**50)
	
#Cumulative Probability of donating in the last x number of days
def CumulativeDonationProbability(x):
	retval=[]
	for i in range(365-x,365):
		retval.append(density[i])
	return(sum(retval))

#Define blood types and compatibility
def Bloodtype():
	ran=random.random()
	if ran<=.374:
		return('O+')
	elif ran<=.374+.357:
		return('A+')
	elif ran<=.374+.357+.085:
		return('B+')
	elif ran<=.374+.357+.085+BloodType[3]:
		return('O-')
	elif ran<=.374+.357+.085+BloodType[3]+BloodType[4]:
		return('A-')
	elif ran<=.374+.357+.085+BloodType[3]+BloodType[4]+BloodType[5]:
		return('AB+')
	elif ran<=.374+.357+.085+BloodType[3]+BloodType[4]+BloodType[5]+BloodType[6]:
		return('B-')
	else:
		return('AB-')


# In[ ]:


#Define Blood applicability (Product can be 'Plasma' or 'RBC')
def CanTransfuse(ApplicantType,Product='RBC'):
	RBCDonors={'O-':['O-'],'O+':['O-','O+'],'A+':['A+','A-','O+','O-'],'A-':['A-','O-'],'B+':['B+','B-','O+','O-'],
			'B-':['B-','O-'],'AB+':['A+','A-','O+','O-','B+','B-','AB+','AB-'],'AB-':['A-','O-','B-','AB-']}
	PlasmaDonors={'O-':['O-','AB-','A-','B-'],'O+':['O-','O+','AB+','AB-','A+','A-','B+','B-'],'A+':['A+','A-','AB+','AB-'],
				  'A-':['A-','AB-'],'B+':['B+','B-','AB+','AB-'],'B-':['B-','AB-'],'AB+':['AB+','AB-'],'AB-':['AB-']}
	if Product=='RBC':
		return(RBCDonors[ApplicantType])
	elif Product=='Plasma':
		return(PlasmaDonors[ApplicantType])
	else:
		return(False)

def AgentsAvailableforTransfusion(ApplicantType,Product='RBC'):
	BloodTypes=CanTransfuse(ApplicantType,Product)
	AgentsAvailable=[]
	for i in BloodTypes: 
		AgentsAvailable += Model.BloodDictionary[i]
	return(AgentsAvailable)

def MostAbundantBloodType(dic,listt):
	dicc = {}
	for i in listt:
		dicc[i]=dic[i]
	return(max(dicc, key= lambda x: len(set(dicc[x]))))
		
TypesofBlood = ['O+','A+','B+','O-','A-','AB+','B-','O-']

def ListDifference(List):
	for i in range(len(List)-1):
		if List[i+1]-List[i] < 55:
			return(False)
	return(True)

def Optomizee(List):
	retval=[]
	for i in range(len(List)-1):
		if List[i+1]-List[i] < 55:
			a = List[i+1]-List[i]
			retval.append(55-a)
	return(sum(retval))

num=[]
n, p, l = 4, 0.6992, 1
for i in range(1,10):    
	num.append(nbinom.pmf(i, n, p,loc=l))
def NumberofDonationss():
	rand=random.random()
	for i in range(1,len(num)+1):
		if rand<=sum(num[0:i]):
			return(i)
			break
		else:
			return(10)

#Insertion Sort algorithm
def ins_sort(k):
	for i in range(1,len(k)):    
		j = i                   
		temp = k[j]             
		while j > 0 and temp < k[j-1]: 
			k[j] = k[j-1] 
			j=j-1 
		k[j] = temp
	return(k)


# In[ ]:


class MoneyModel(Model):    
	def __init__(self, N):
		self.AgentswithPlatelets = []
		self.outdatedRBC = 0
		self.outdatedPlatelet = 0
		self.BloodDictionary={'O-':[],'O+':[],'A+':[],'A-':[],'B+':[],'B-':[],'AB+':[],'AB-':[]}
		self.HasRBC={'O-':[],'O+':[],'A+':[],'A-':[],'B+':[],'B-':[],'AB+':[],'AB-':[]}
		self.HasPlasma={'O-':[],'O+':[],'A+':[],'A-':[],'B+':[],'B-':[],'AB+':[],'AB-':[]}
		self.num_agents = N
		self.Day = 0
		self.schedule = RandomActivation(self)
		# Create agents
		print('Generating Agents...')
		for i in tqdm(range(self.num_agents)):
			a = MoneyAgent(i, self, self.Day)
			a.ID = i
			self.schedule.add(a)
			if a.Elig and not a.HIV:
				self.BloodDictionary[a.Blood].append(a)
		j = 0
		for i in range(int(357000/315000000*self.num_agents)):
			TimeCalculation = int(357000/315000000*self.num_agents/42)
			Bloodtype = np.random.choice(TypesofBlood,p=BloodType)
			Agent = random.choice(self.BloodDictionary[Bloodtype])
			Agent.RBCBank += 1
			Agent.PlasmaBank += 1
			self.HasRBC[Bloodtype].append(Agent)
			self.HasPlasma[Bloodtype].append(Agent)
			if i%TimeCalculation==0:
				j += 1
			Agent.RBCTime = j
		j = -1
		for i in range(int(70200/315000000*self.num_agents)):
			TimeCalculation = int(70200/315000000*self.num_agents/5)
			Agent = random.choice(self.BloodDictionary[random.choice(TypesofBlood)])
			Agent.PlateletBank += 1
			if i%TimeCalculation == 0:
				j += 1
			Agent.PlateletTime = j
			self.AgentswithPlatelets.append(Agent)
		#for i in range(int(300000/315000000*self.num_agents)):
		#	Bloodtype = np.random.choice(TypesofBlood,p=BloodType)
		#	Agent = random.choice(self.BloodDictionary[Bloodtype])
		#	Agent.PlasmaBank += 1
		#	self.HasPlasma[Bloodtype].append(Agent)
		self.datacollector = DataCollector(
			model_reporters = {'TotalHivBloodProducts': TotalHivBloodProducts, 'test':test,"HIVPrevalence": computeHIVPrevalence, 'FailDonor': FailedHIVDonors, 'PlasmabyType': PlasmabyType, 'HIVDonate':HIVDon,"RBCbyType":BloodbyType, 
			"Outdated":OutdatedStocks,'TotalDonations':TotalDonations, 'InfectionDays':InfectionDays, 'HIVDonors':HIVDonors})
		
	def step(self):
		self.datacollector.collect(self)
		self.Day += 1      
		self.schedule.step()


# In[ ]:


Plasmaperperson = 2
Plateletsperperson = 2

class MoneyAgent(Agent):

	def __init__(self, unique_id, model, Day):
		super().__init__(unique_id, model)
		#Assign bloodtype, sexuality, HIV status, and other default properties to Agents
		self.ID = 0
		self.Blood = Bloodtype()
		MSMrand =    random.random()
		HIVrand =    random.random()
		CancerAnemiarand = random.random()
		self.MSM = False 
		self.HIV = False
		self.clueless = True
		self.Dayssinceinfection = 0
		self.RBCBank,self.PlateletBank,self.PlasmaBank = [0,0,0]
		self.WillDonate = False
		self.Donationday = []
		self.totPlat,self.totRBC = [0,0]
		self.Day = Day
		self.RBCTime,self.PlateletTime = [0,0]
		self.HIVDetected = False
		self.HIVDonation = False
		self.ReceivedHIV = False
		self.EndStageRenalDisease = False
		self.FalsePositive = False
		self.Donations = 0
		self.PlateletsHIV = 0
		self.RBCHIV = 0
		self.PlasmaHIV = 0
		if MSMrand <= PMSM:
			self.MSM = True
		if HIVrand <= PHIV[0] and self.MSM and Day == 0:
			self.HIV = True
		elif HIVrand <= PHIV[1] and not self.MSM and Day == 0:
			self.HIV = True        
		
		#Determine Agent eligibility (MSM excluded)
        if ExcludeMSM and self.MSM:
            self.Elig = False
		if not self.clueless:
			self.Elig = False
		elif self.MSM:
			self.Elig = False
		elif random.random() <= Eligibility:
			self.Elig = True
		else:
			self.Elig = False
			
		if self.HIV:  
			dayrand = random.random()
			if dayrand >= MassIncidence(self.MSM):
				self.Dayssinceinfection = 51
				if random.random()<=.85:
					self.clueless = False
			else:
				self.Dayssinceinfection = random.randint(1,50)
				self.clueless = True
		#Probability of Actually Donating	
		if self.Elig and random.random()<=.065:
			self.WillDonate = True
			respectiveprobabilities = [0.273655, 0.32976, 0.24732, 0.14985]
			rand = random.random()
			for n,i in enumerate(respectiveprobabilities):
				if rand<=sum(respectiveprobabilities[0:n+1]):
					numbsamples = n + 1
					break
			self.Donationday = ins_sort(list(sample_iter(X,numsamples=numbsamples)))
			optim = self.Donationday
			if len(self.Donationday) > 1:
				count = 0
				while not ListDifference(self.Donationday):
					count += 1
					self.Donationday = ins_sort(list(sample_iter(X,numsamples=numbsamples)))
					if Optomizee(self.Donationday) < Optomizee(optim):
						optim = self.Donationday
					if count == 5:
						self.Donationday = optim
						break
			 
	def DonateWB(self):
		self.Time = 0
		self.RBCTime,self.PlateletTime = [0,0]
		self.RBCBank += 1
		self.PlateletBank += 0.2
		self.PlasmaBank += 1
		self.totPlat += .2
		self.totRBC += 1
		Model.AgentswithPlatelets = [self] + Model.AgentswithPlatelets
		Model.HasRBC[self.Blood] = [self] + Model.HasRBC[self.Blood]
		Model.HasPlasma[self.Blood] = [self] + Model.HasPlasma[self.Blood]
		
	def DonatePlatelets(self):
		self.Time = 0
		self.PlateletTime = 0
		self.PlateletBank += 1.8
		self.totPlat += 1
		Model.AgentswithPlatelets = [self] + Model.AgentswithPlatelets
		
	def DonateRBC(self):
		self.Time = 0
		self.RBCTime = 0
		self.RBCBank += 2
		self.totRBC += 2
		Model.HasRBC[self.Blood] = [self] + Model.HasRBC[self.Blood]
 
	def Donation(self, Day):
		self.RBCTime += 1
		self.PlateletTime += 1    
		if Day in self.Donationday and not self.HIV:
			f=time.time()
			self.Donations += 1
			if random.random() <= (1-Specificity):
				self.FalsePositive = True
				self.Donationday = []
			else:
				rand=random.random()
				if rand<=.0684:
					self.DonateRBC()
				elif rand<=(.8423+.0684):
					self.DonateWB()
				else:
					self.DonatePlatelets()
			#TimingDonation.append(time.time()-f)
		elif Day in self.Donationday and self.HIV:
			self.HIVBloodDonations()
 
	def acquireHIV(self):
		g=time.time()   
		if self.HIV and self.Dayssinceinfection < 51:
			self.Dayssinceinfection += 1
		if self.MSM and not self.HIV:
			if random.random() <= AcHIV[0]:
				self.HIV = True
				self.clueless = True
		if not self.MSM and not self.HIV:
			if random.random() <= AcHIV[1]:
				self.HIV = True
				self.clueless = True
		#TimingHIV.append(time.time()-g)
			 
	def HIVBloodDonations(self):
		if random.random() <= FalseNegative(self.Dayssinceinfection):
			self.Donations += 1
			rand=random.random()
			self.HIVDonation = True
			if rand<=.0684:
				self.DonateRBC()
			elif rand<=.8423+.0684:
				self.DonateWB()
			else:
				self.DonatePlatelets()
		else:
			self.Elig = False
			self.WillDonate = False
			self.clueless = False
			self.Donationday = []
			self.HIVDetected = True                                   
	   
	def Outdated(self):
		h=time.time()
		if self.RBCTime >= RBCDecay:
			Model.outdatedRBC += self.RBCBank
			self.RBCBank = 0
			self.RBCTime = 0
		if self.PlateletTime >= PlateletDecay:
			Model.outdatedPlatelet += self.PlateletBank
			self.PlateletTime = 0
			self.PlateletBank = 0
		if self.PlateletBank <= 0 and self in Model.AgentswithPlatelets:
			Model.AgentswithPlatelets.remove(self)
		if self.RBCBank < 1 and self in Model.HasRBC[self.Blood]:
			Model.HasRBC[self.Blood].remove(self)                 
		if self.PlasmaBank < 1 and self in Model.HasPlasma[self.Blood]:
			Model.HasPlasma[self.Blood].remove(self) 
		#TimingOutdated.append(time.time()-h)

	def ReceivedHIVTransfusion(self):
		if random.random() <= .9:
			self.HIV = True
			self.ReceivedHIV = True

	def GeneralizedPlateletsUsed(self):
		if random.random() <= (2200000/Plateletsperperson)/315000000/365:
			self.ExtractPlatelets(Plateletsperperson)

	def GeneralizedRBCUsed(self):
		if random.random() <= .0146/365:
			if random.random() <= .28:
				self.ExtractRBC(2,ProbOfNoCrossmatch=1)
			else:
				self.ExtractRBC(3,ProbOfNoCrossmatch=1)

	def GeneralizedPlasmaUsed(self):
		if random.random() <= (3300000/Plasmaperperson)/315000000/365:
			self.ExtractPlasma(Plasmaperperson)

	def ExtractPlatelets(self,BloodNeeded):
		c=time.time()
		BloodDrawn = 0
		rangge = range(-1,-1*len(Model.AgentswithPlatelets),-1)
		while BloodDrawn < BloodNeeded:
			if Model.AgentswithPlatelets==[]:
				return(print('Platelet Shortage'))    
			Donor = Model.AgentswithPlatelets[-1]
			if 0 < Donor.PlateletBank < 1:
				BloodDrawn += Donor.PlateletBank
				Donor.PlateletBank = 0
				del Model.AgentswithPlatelets[-1]
				if Donor.HIVDonation and Donor.Dayssinceinfection > Donor.PlateletTime:
					Donor.PlateletsHIV += 1
					self.ReceivedHIVTransfusion()
			elif Donor.PlateletBank >= 1: #and (BloodNeeded-BloodDrawn) >= 1:
				Donor.PlateletBank -= 1
				BloodDrawn += 1
				if Donor.PlateletBank <= 0:
					del Model.AgentswithPlatelets[-1]
				if Donor.HIVDonation and Donor.Dayssinceinfection > Donor.PlateletTime:
					Donor.PlateletsHIV += 1
					self.ReceivedHIVTransfusion()
			else:
				continue
		#TimingPlatelets.append(time.time()-c)
							
	def ExtractRBC(self,BloodNeeded,ProbOfNoCrossmatch = 1):
		d=time.time()
		quit = 0
		Donor = random.choice(Model.AgentswithPlatelets)
		lists = CanTransfuse(self.Blood)
		Bloodd = MostAbundantBloodType(Model.HasRBC,lists)
		if random.random() <= ProbOfNoCrossmatch:
			BloodDrawn = 0
			quit = 0
			a=time.time()
			while BloodDrawn < BloodNeeded:
				while not Model.HasRBC[Bloodd]:
					Bloodd = random.choice(lists)
					quit += 1
					if quit == 10:
						return(print(['RBC Shortage',self.Blood]))
						print(time.time()-a)
				Donor = Model.HasRBC[Bloodd][-1]
				if Donor.RBCBank >= 1:
					Donor.RBCBank -= 1
					BloodDrawn += 1
					if Donor.HIVDonation and Donor.Dayssinceinfection > Donor.RBCTime:
						Donor.RBCHIV += 1
						self.ReceivedHIVTransfusion()
				if Donor.RBCBank < 1:
					Donor.RBCBank = 0
					del Model.HasRBC[Bloodd][-1]
		else:
			BloodDrawn = 0
			while BloodDrawn < BloodNeeded:
				if Model.HasRBC['O-'] == []:
					return(print(['RBC Shortage',self.Blood]))
				Donor = Model.HasRBC['O-'][-1]
				if Donor.RBCBank >= 1:
					Donor.RBCBank -= 1
					BloodDrawn += 1
					if Donor.HIVDonation and Donor.Dayssinceinfection > Donor.RBCTime:
						Donor.RBCHIV += 1
						self.ReceivedHIVTransfusion()
				if Donor.RBCBank < 1:
					Donor.RBCBank = 0
					del Model.HasRBC['O-'][-1]
		#TimingRBC.append(time.time()-d)

	def ExtractPlasma(self,BloodNeeded,ProbOfNoCrossmatch = 1):
		e=time.time()
		quit = 0
		Donor = random.choice(Model.schedule.agents)
		lists = CanTransfuse(self.Blood,Product='Plasma')
		Bloodd = MostAbundantBloodType(Model.HasPlasma,lists)
		#print(Model.HasPlasma)
		if random.random() <= ProbOfNoCrossmatch:
			BloodDrawn = 0
			a=time.time()
			quit=0
			while BloodDrawn < BloodNeeded:
				while not Model.HasPlasma[Bloodd]:
					Bloodd = random.choice(lists)
					quit += 1
					if quit == 10:
						return(print(['Plasma Shortage',self.Blood]))
						print(time.time()-a)
				Donor = random.choice(Model.HasPlasma[Bloodd])
				if Donor.PlasmaBank < 1:
					Donor.PlasmaBank = 0
					Model.HasPlasma[Bloodd].remove(Donor)
				elif Donor.PlasmaBank >= 1:
					Donor.PlasmaBank -= 1
					BloodDrawn += 1
					if Donor.HIVDonation and Donor.Dayssinceinfection > Donor.RBCTime:
						Donor.PlasmaHIV += 1
						self.ReceivedHIVTransfusion()
		else:
			BloodDrawn = 0
			while BloodDrawn < BloodNeeded:
				if Model.HasPlasma['AB-'] == []:
					return(print(['Plasma Shortage',self.Blood]))
				Donor = random.choice(Model.HasPlasma['AB-'])
				if Donor.PlasmaBank < 1:
					Donor.PlasmaBank = 0
					Model.HasPlasma[Bloodd].remove(Donor)
				elif Donor.PlasmaBank >= 1:
					Donor.PlasmaBank -= 1
					BloodDrawn += 1
					if Donor.HIVDonation and Donor.Dayssinceinfection > Donor.RBCTime:
						Donor.PlasmaHIV += 1
						self.ReceivedHIVTransfusion()
		#TimingPlasma.append(time.time()-e)


	def step(self):
		self.Outdated()
		self.Day += 1
		self.Donation(self.Day)
		self.acquireHIV()
		self.GeneralizedPlasmaUsed()
		self.GeneralizedRBCUsed()
		self.GeneralizedPlateletsUsed()


# In[ ]:


agentnum = 200000
Plasmaperperson = random.randint(1,4)
Plateletsperperson = random.randint(1,4)
PMSM = random.uniform(.02,.05)
MSMandHIV = 0.001917
NotMSMandHIV = 0.001509
MSMacquire =  26200/(321039839*PMSM*(1-MSMandHIV/PMSM))
NotMSMacquire = 8800/(321039839*(1-PMSM))
PHIV=[MSMandHIV/PMSM,NotMSMandHIV/(1-PMSM)]
AcHIV=[MSMacquire/365,NotMSMacquire/365]
if ExcludeMSM:
    Eligibility= .38/(1-PMSM)
else:
    Eligibility= .38 + PMSM*(1-PHIV[0])
a=time.time()
Model = MoneyModel(agentnum)
print('Running Model')
for i in tqdm(range(365)): 
	Model.step()
data = Model.datacollector.get_model_vars_dataframe()
b=time.time()
print(str((b-a)/60)+' Minutes Elapsed')

Date= str(random.randint(1,9999999))

file_Name = Date+'HIVDetected'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump([a.HIVDetected for a in Model.schedule.agents],fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'HIVDonations'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump([a.HIVDonation for a in Model.schedule.agents],fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'HIV'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump([a.HIV for a in Model.schedule.agents],fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'HIVReceived'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump([a.ReceivedHIV for a in Model.schedule.agents],fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'Outdated'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.Outdated,fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'Total'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.TotalDonations,fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'TotalHIVDonations'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.HIVDonate,fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'MSM'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump([a.MSM for a in Model.schedule.agents],fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'RBCbyType'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.RBCbyType,fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'FalsePositive'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump([a.FalsePositive for a in Model.schedule.agents],fileObject, protocol=2)   
fileObject.close()
file_Name = Date+'RBCbyType'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.RBCbyType,fileObject, protocol=2) 
fileObject.close()
file_Name = Date+'InfectionDays'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.InfectionDays,fileObject, protocol=2) 
fileObject.close()
file_Name = Date+'HIVDonors'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.HIVDonors,fileObject, protocol=2)
fileObject.close()
file_Name = Date+'FailedHIVDonors'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.FailDonor,fileObject, protocol=2) 
fileObject.close()
file_Name = Date+'HIVBloodProducts'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.test,fileObject, protocol=2) 
fileObject.close()
file_Name = Date+'PerPerson'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump([Plasmaperperson,Plateletsperperson],fileObject, protocol=2) 
fileObject.close()
file_Name = Date+'PlasmabyType'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.PlasmabyType,fileObject, protocol=2) 
fileObject.close()
file_Name = Date+'TotalHivBloodProducts'+'.pickle'
fileObject = open(file_Name,'wb') 
pickle.dump(data.TotalHivBloodProducts,fileObject, protocol=2) 
fileObject.close()
del(data)
del(Model)

