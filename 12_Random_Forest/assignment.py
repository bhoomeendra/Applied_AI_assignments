import numpy as np
from sklearn.datasets import load_boston 
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

class RegressionRandomForest():# Can make it such that you can pass an array of models a general bagging frame_work 

    
    def __init__(self,n_estimator,ratio_unique,min_col) -> None:
        self.n_estimator  = n_estimator
        self.ratio_unique = ratio_unique
        self.min_col = min_col # for forming a bootstaped sample
        self.input_row = None
        self.input_cols = None
        self.models = None
        self.rows = None
        self.cols = None
        self.unique_pt = None
        self.mse = None
        self.oob_score = None

    def fit(self,x,y):
        self.input_row = x.shape[0]
        self.input_cols = x.shape[1]
        self.unique_pt = int(self.input_row*self.ratio_unique)
        print(self.input_row,self.input_cols,self.unique_pt)
        self.rows = [] # this will store the indexs of the samples that are bootstraped from the data
        self.cols = [] # this will store the cols that are selected and both will be in the same order
        self.models = []
        total_pred = np.zeros(self.input_row)
        total_oob = np.zeros(self.input_row)
        count_oob_update = np.zeros(self.input_row)# Count maintains how many times a point has updated
        for i in tqdm(range(self.n_estimator)) :
            row,col = self.genrate_sample()
            self.rows.append(row)
            self.cols.append(col)
            model = DecisionTreeRegressor()
            model.fit(x[row,:][:,col],y[row])
            total_pred +=  model.predict(x[:,col])
            oob_idx  = [ i for i in range(self.input_row) if i not in row ]
            total_oob[oob_idx] += model.predict(x[oob_idx,:][:,col])
            count_oob_update[oob_idx] += 1 
            self.models.append(model)
        mean_oob_pred = total_oob/count_oob_update
        mean_pred = total_pred/self.n_estimator
        self.oob_score = mean_squared_error(mean_oob_pred,y)
        self.mse = mean_squared_error(mean_pred,y)
    
    def precit(self,x):
        total_pred = np.zeros(x.shape[0])# Number of outputs
        for i in range(self.n_estimator) :
            #pdb.set_trace()
            total_pred += self.models[i].predict(x[:,self.cols[i]]) 
        return total_pred/self.n_estimator

    def genrate_sample(self):
        selecting_rows = np.random.choice(self.input_row, self.unique_pt ,replace = False)# No repetation 
        replacing_rows = np.random.choice(selecting_rows,self.input_row - self.unique_pt,replace = True)# This can be repeating
        selecting_col = np.random.choice(self.input_cols,np.random.randint(self.min_col,self.input_cols),replace = False)
        return np.hstack((selecting_rows,replacing_rows)) , selecting_col



if __name__ == '__main__':
    boston = load_boston()
    x=boston.data #independent variables
    y=boston.target #target variable
    #x=pd.DataFrame(x)
    print("#"*40 + "\n"*10)
    model = RegressionRandomForest(n_estimator=30,ratio_unique=0.6,min_col=3)
    model.fit(x=x,y=y)
    print("THE MSE IS : ",model.mse)
    print("THE oob_score IS : ",model.oob_score)
    xq = np.array([0.18,20.0,5.00,0.0,0.421,5.60,72.2,7.95,7.0,30.0,19.1,372.13,18.60])
    print("The prediction: ",model.precit(xq.reshape(1,-1)))
    