'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
    entropy = 0
    value, counts = np.unique(df.iloc[:,-1], return_counts=True)
    norm_counts = counts / counts.sum()
    base = 2
    entropy = -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
    return entropy

'''
Assignment1 code
Team name: PES-MI_0254_2029


'''

'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
    entropy_of_attribute = 0
    value, counts = np.unique(df[attribute], return_counts=True)
    norm_counts = counts / counts.sum()
    base = 2
    entropy = np.array([])
    for i in range(0,len(value)):
        split = df.loc[df[attribute] == value[i]]
        val, cnts = np.unique(split.iloc[:,-1], return_counts=True)
        norm_cnts = cnts / cnts.sum()
        entropy = np.append(entropy,-(norm_cnts * np.log(norm_cnts) / np.log(base)).sum())
    entropy_of_attribute = (norm_counts * entropy).sum()
    return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = 0
	entropy=get_entropy_of_dataset(df)
	average_information=get_entropy_of_attribute(df,attribute)
	information_gain=entropy-average_information
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):

	'''
	Return a tuple with the first element as a dictionary which has IG of all columns
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

	information_gains={}
	selected_column=''
	for attribute in df.columns[:-1]:
		information_gains[attribute]=get_information_gain(df,attribute)

	for attribute in df.columns:
		if information_gains[attribute]==max(information_gains.values()):
			selected_column=attribute
			break


	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''