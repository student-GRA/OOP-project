import  argparse
import pandas as pd
import statsmodels.api as sm
from glm_ex import Normal, Poisson, Bernoulli
from data_loaders import sm_loader, csv_loader


# Adds the user interface. Create choices for input.
parser = argparse.ArgumentParser(description='Test if GLM objects have the same outputs as the sm.GLM')
parser.add_argument('--model', choices=['normal', 'bernoulli', 'poisson'], default = 'normal')
#parser.add_argument('--dset', type=str , default ='duncan' ,  help='which dataset to use for testing', choices=['duncan', 'spector', 'warpbreaks'])
parser.add_argument('--predictors', nargs='+' , default ='all' ,  help='which predictors to use. Use column names')
parser.add_argument('--add-intercept', action='store_true')
#parser.add_argument('--transpose', action='store_true')
args = parser.parse_args()


# Choose inputs based on command line choices.
model_select = {'normal': [Normal,sm.families.Gaussian(), 'duncan'], 'poisson': [Poisson, sm.families.Poisson(), 'warpbreaks'], 'bernoulli': [Bernoulli, sm.families.Binomial(), 'spector']}
c, sm_family, dset = model_select[args.model]
csv_file = {'warpbreaks': 'https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv'}

print(f'Will be using the {dset} dataset to compare the {args.model} models')

# load correct dataset.
if dset in ['spector', 'duncan']:
    loader = sm_loader()
    loader.load_data(dset)
else:
    loader = csv_loader()
    loader.load_data(csv_file[dset])

# Assign x's and y's
loader.assign_x_y(args.predictors)

#Add intercept if chosen in command line      
if args.add_intercept == True:
    loader.add_intercept()

#if args.transpose == True:
    #loader.x_transpose()

# Get x and y from loader. Train my class model and the statsmodels model.
x = loader.x
y = loader.y
model = c()
model = model.fit(x,y)
m = sm.GLM(y,x, family = sm_family)
res = m.fit()
params = res.params


# Print the betas
print(f'My {args.model} class produced the beta values:')
print(model.betas)

print(f'The statsmodels GLM equivilant produced:')
print(list(params))


# Check if they are the same down to the 4th decimal.
if [round(el,4) for el in model.betas] == [round(el,4) for el in params]:
    print('These estimates are equal down to the 4th decimal point.')
else:
    print('These estimates deviate.')

print(f'{'-'*60}')

cl_pred = model.predict(x)
sm_predict = res.predict(x)

# Check if the predictions are equal down to the third decimal
if [round(el,3) for el in cl_pred] == [round(el,3) for el in sm_predict]:
    print('The predictions are equal down to the 3rd decimal point')




