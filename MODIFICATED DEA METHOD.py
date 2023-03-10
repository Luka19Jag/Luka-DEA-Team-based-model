import numpy as np
import pandas as pd
#from scipy.sparse import csr_matrix#csr_array
from scipy.optimize import linprog

data = pd.read_csv('D:/Luka Sve/Strucni radovi/Modelling DEA mathematical model on athlete\'s efficiency score with the influence of the team/final_stats.csv')
data = data.set_index('player_name')

#xal = csr_matrix(data)
#yal = csr_array()

input = data.columns[2:8]
input = input.append(data.columns[0:1]) 
output = data.columns[8:]


X = data[input]
team = X['club']
max_values_X = X.max()[:-1]
X = X.loc[:, (X.columns != 'club')]/max_values_X
X['club'] = team
Y = data[output]
max_values_Y = Y.max()
Y = Y / max_values_Y

epsilon = 0.00000001
gamma = 0.85 

inputs_attr = len(input)-1 # -1 is because of club, it is not calculated here
outputs_attr = len(output) 

number_of_all_inputs = inputs_attr * len(data.index) 
number_of_all_outputs = outputs_attr * len(data.index) 

   #--- DICTIONARY OF ALL CLUBS ---
dic_club = {}
for i in range(X.shape[0]):
    if(X.iloc[i,:]['club'] in dic_club):
        dic_club[X.iloc[i,:]['club']]['Number'] = dic_club[
            X.iloc[i,:]['club']]['Number'] + 1
    else:
        dic_club[X.iloc[i,:]['club']] = {'Number':1}


   #--- VARIABLES ---
w = np.repeat(0.0, number_of_all_inputs + number_of_all_outputs)

   #--- GOAL ---
goal = np.repeat(0.0, number_of_all_inputs + number_of_all_outputs)

for i in range(X.shape[0]):
    start_input = i * inputs_attr
    end_input = i * inputs_attr + inputs_attr - 1
    
    goal[start_input:(end_input + 1)] = X.iloc[i, :-1] # -1 for club
    
   #--- VIRTUAL OUTPUT HAS TO BE 1 --- FIRST CONSTRAINT
Aeq = []
Beq = [1]*(X.shape[0])

for i in range(X.shape[0]):
    i_Aeq = np.repeat(0.0, number_of_all_inputs + number_of_all_outputs)
    
    start_output = number_of_all_inputs + i * outputs_attr
    end_output = number_of_all_inputs + i * outputs_attr + outputs_attr - 1
    
    i_Aeq[start_output:(end_output + 1)] = Y.iloc[i, :]
    
    Aeq.append(i_Aeq)
    
   #--- VIRTUAL OUTPUT MINUS VIRTUAL INPUT HAS TO BE LESS THAN 0 --- SECOND
   # CONSTRAINT
A = []
B = np.repeat(0.0, X.shape[0]*X.shape[0])
for i in range(X.shape[0]):
    start_input = i * inputs_attr
    end_input = i * inputs_attr + inputs_attr - 1
    
    start_output = number_of_all_inputs + i * outputs_attr
    end_output = number_of_all_inputs + i * outputs_attr + outputs_attr - 1
    
    for j in range(X.shape[0]):
        ijA = np.repeat(0.0, number_of_all_inputs + number_of_all_outputs)
        ijA[start_input:(end_input + 1)] = -X.iloc[j, :-1] # -1 for club
        
        ijA[start_output:(end_output + 1)] = Y.iloc[j, :]
        
        A.append(ijA)

   #--- LOWER BOUND WHEN TEAM INFLUENCE ON PLAYER --- NEW THIRD CONSTRAINT
   #--- LOWER BOUND FOR INPUT => UPPER BOUND FOR PLAYER'S EFFICIENCY
B = np.append(B, [0]*X.shape[0]) # adding 0 for every player

for i in range(X.shape[0]):
    ij_dummy = np.repeat(0.0, number_of_all_inputs + number_of_all_outputs)
    for j in range(X.shape[0]):
        start_input = j * inputs_attr
        end_input = j * inputs_attr + inputs_attr - 1
        if(i == j):
            ij_dummy[start_input:(end_input + 1)] = -(
                dic_club[X.iloc[i,:]['club']][
                    'Number'] - gamma)/dic_club[
                    X.iloc[i,:]['club']]['Number']*X.iloc[j,:-1]
            # minus is putted because the condition is <=0
        elif(X.iloc[i,:]['club'] != X.iloc[j,:]['club']):
            ij_dummy[start_input:(end_input + 1)] = X.iloc[j,:-1]*0
            # write all 0
        else:
            ij_dummy[start_input:(end_input + 1)] = gamma*X.iloc[
                j,:-1]/dic_club[X.iloc[i,:]['club']]['Number']
            # minus is putted because the condition is <=0
    A.append(ij_dummy) # immediately is added on matrix A

   #--- UPPER BOUND WHEN TEAM INFLUENCE ON PLAYER --- NEW FOURTH CONSTRAINT
   #--- UPPER BOUND FOR INPUT => LOWER BOUND FOR PLAYER'S EFFICIENCY
B = np.append(B, [0]*X.shape[0]) # adding 0 for every player

for i in range(X.shape[0]):
    ij_dummy = np.repeat(0.0, number_of_all_inputs + number_of_all_outputs)
    for j in range(X.shape[0]):
        start_input = j * inputs_attr
        end_input = j * inputs_attr + inputs_attr - 1
        if(i == j):
            ij_dummy[start_input:(end_input + 1)] = (gamma*
                dic_club[X.iloc[i,:]['club']]['Number'] - 1)/(dic_club[
                    X.iloc[i,:]['club']]['Number']*gamma)*X.iloc[j,:-1]
        elif(X.iloc[i,:]['club'] != X.iloc[j,:]['club']):
            ij_dummy[start_input:(end_input + 1)] = X.iloc[j,:-1]*0
            # write all 0
        else:
            ij_dummy[start_input:(end_input + 1)] = -1/(gamma*dic_club[
                X.iloc[i,:]['club']]['Number'])*X.iloc[j,:-1]
            # minus is putted because the condition is <=0
    A.append(ij_dummy) # immediately is added on matrix A


   #--- MODEL FOR LEARNING ---
bnds = [(epsilon, float("inf"))] * len(w)

res = linprog(goal, 
              A_ub=A, b_ub=B, 
              A_eq=Aeq, b_eq=Beq, 
              bounds=bnds)
res.x
slack = res.slack

   #--- PLAYER'S EFFICIENCY --- 
dic_vir_attr = {}
dic_efficiency = {}
dic_role_models = {}
for i in range(X.shape[0]):
    start_input = i * inputs_attr
    end_input = i * inputs_attr + inputs_attr - 1
    
    start_output = number_of_all_inputs + i * outputs_attr
    end_output = number_of_all_inputs + i * outputs_attr + outputs_attr - 1
    
    inputs = res.x[start_input:(end_input + 1)]
    outputs = res.x[start_output:(end_output + 1)]
    
    Hk = np.sum(res.x[start_input:(end_input + 1)]*X.iloc[i,:-1])
    Gk = 1/Hk
    
    if(Gk > 0.99999):
        Gk = 1
    
    i_slack = i*X.shape[0]+i
    slack = res.slack[i_slack]
    
    if(slack < 0.00001):
        slack = 0
        dic_role_models[X.iloc[i,:].name] = {'Role_model':'/'}
    else:
        start_slack = i*X.shape[0]
        end_slack = i*X.shape[0] + X.shape[0]
        result = np.where(res.slack[start_slack:end_slack] <= 0.00000000001)
        index = np.where(res.slack[
            start_slack:end_slack].min() == res.slack[start_slack:end_slack])
        names = X.iloc[index[0][0],:].name + ', '
        result = np.delete(result, np.where(result[0] == index[0][0]))
        
        for role_model in range(len(result)):
            if(role_model == (len(result)-1)):
                names += X.iloc[result[role_model],:].name
            else:
                names += X.iloc[result[role_model],:].name + ', '
        
        dic_role_models[X.iloc[i,:].name] = {'Role_model':names}
    
    dic_efficiency[X.iloc[i,:].name] = {
        'Index':Hk, 'Efficiency':Gk, 'Slack': slack}
    dic_vir_attr[X.iloc[i,:].name]  = {
        'minutes_played':inputs[0], 'shots_on_target':inputs[1], 
        'dribbles': inputs[2], 'pass_completed':inputs[3],
        'tackles_won':inputs[4], 'clearance_attempted':inputs[5] ,
        'goals':outputs[0] ,'assists': outputs[1],'balls_recoverd':outputs[2]}

# normalized data
data = pd.concat([X,Y], axis=1) 
# virtual variables on normalized data
weights_norm = pd.DataFrame(dic_vir_attr).T * data.loc[
    :,(data.columns != 'club')]

# denormalization data
X = X.loc[:,(X.columns != 'club')]*max_values_X
Y = Y * max_values_Y
data = pd.concat([X,Y], axis=1)
# virtual variables on denormalized data
weights = pd.DataFrame(dic_vir_attr).T * data.loc[
    :,(data.columns != 'club')]

dic_vir_attr = pd.DataFrame(dic_vir_attr).T
dic_role_models = pd.DataFrame(dic_role_models).T
dic_efficiency = pd.DataFrame(dic_efficiency).T


# Nisi nista sacuvao !!!

#df_efficiency.to_csv('C:/Users/Zbook G3/Desktop/df_efikasnost_modifikovani_CCR2.csv')
#weights_norm.to_csv('C:/Users/Zbook G3/Desktop/weights_norm__modifikovani_CCR2.csv')
#df_uzori.to_csv('C:/Users/Zbook G3/Desktop/df_uzori_modifikovani_CCR2.csv')

slack_for_active_constraints = res.slack
