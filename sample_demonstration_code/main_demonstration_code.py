#%%
import pandas as pd
import func_ATMD_Geo
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.util import load_logs
import pandas as pd
import math
import os
import numpy as np
from scipy.optimize import NonlinearConstraint
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sko.GA import GA

'''
this is an example of structure control and floating platform geometry optimization framework.

include:
    parameters boundary, 
    objective function,
    constraint function,
    data flow transmission, 
    optimization loop,
    data output
    ...
'''
#%%

bound={                    ## Boundary of the parameters     
    "columns_D":(9,11),
    'TMD_mass':(18674,186740),
    'TMD_stiffness':(9000,1.2e6), 
}

col=[
    "columns_D",
    "TMD_mass",
    "TMD_stiffness",
    "Platform_ultPitch",
    "AEP",
]

##    define the objective function for optimization  
def obj_fun(pitch):
    value=-pitch  
    return value

def constraint_advanced(AEP):
    return AEP
constraint_lower=np.array([3.95e7])
constraint_upper=np.array([np.inf])

constraint=NonlinearConstraint(constraint_advanced,constraint_lower,constraint_upper)
optimizer=BayesianOptimization(   
    f=None,
    pbounds=bound,
    constraint=constraint,
    verbose=2,
    random_state=1,
    allow_duplicate_points=True
    )
utility=UtilityFunction('ucb',kappa=2.576,xi=0.0)

data=pd.DataFrame(columns=col)   ## main data flow definition


for i in range(2):
    print('NO.'+str(i)+' sample simulation start!!!')

    next_point=optimizer.suggest(utility)
    point_frame=pd.DataFrame(next_point,index=[i],columns=data.columns)
    data.loc[i]=point_frame.loc[i].copy()
    para=data.loc[i].copy()

    num=str(i)

    flag_AQWA=func_ATMD_Geo.AQWA_run_sample(num,para)  # flag_AQWA=1 AQWA run successfully 

    if flag_AQWA==1:
        print("AQWA simulation complete!!!")
    else:
        print('AQWA calcualtion fail.')


    #% transform .AH1 file
    flag_tran=func_ATMD_Geo.transform_AH(num)
    if flag_tran==1:
        print('Hydrodynamic file translation complete!!!')
    else:
        print('Hydrodynamic file translation fail.')

    #%  Openfast run
    para,flag_cp,flag_op,flag_postop,flag_650=func_ATMD_Geo.Openfast_run_sample(num,para)

    target=obj_fun(para['Platform_ultPitch'])
    constraint_value=constraint_advanced(para['AEP'])

    optimizer.register(params=next_point,target=target,constraint_value=constraint_value)

    data.loc[i]=para.copy()
    data.to_excel('data_AGopt.xlsx',sheet_name='1',index=True)
#%%
