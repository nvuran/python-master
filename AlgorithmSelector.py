import datetime
from xgboost import XGBRFRegressor
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import  PolynomialFeatures
from catboost import CatBoostRegressor
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
class AlgorithmSelector:

    def __init__(self,data,testSize,yColumnName):
        self.firstScoreTable = pd.DataFrame()
        self.type = "Classification"
        self.names = []
        self.distances = []
        self.algorithms = []
        self.sqrts = []
        self.testSize = testSize
        self.yColumnName = yColumnName
        self.data = data.dropna()
        self.x = self.data.drop([yColumnName], axis=1)
        self.y = self.data[yColumnName]
        self.X = self.x.values
        self.Y = self.y.values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=testSize, random_state=0)
        self.resultDf = pd.DataFrame()
        self.tableHeaders = self.x.columns
        self.confusionMatrixes = [];
        self.f1Scores = [];
        self.precisionScores = [];
        self.recallScores = [];
        self.macroAvgs=[];
        self.microAvgs = [];
    def getInfo(self,index):
        y_pred = self.algorithms[index].predict(self.x_test)
        y_test = pd.DataFrame(data=self.y_test,columns=['YTest'])
        y_pred = pd.DataFrame(data=y_pred,columns=['YPred'])
        result = pd.concat([y_test, y_pred], axis=1)
        result.columns = ['YTest', 'YPred']
        result['Fark'] = abs(result['YTest'] - result['YPred'])
        model = sm.OLS(y_pred, self.x_test)
        return model.fit().summary()

    def deleteWorstPvalues(self,count,resultStr):
        s = str(resultStr)
        c = 1285
        writingStr = "index,coef,stderr,t,P>|t|,[0.025,0.975]\n"
        while s[c] != '=':
            writingStr += s[c]
            c = c + 1
        splitableString = ""
        spaceState = False
        for i in range(0, len(writingStr)):
            if writingStr[i] == " " and spaceState == False:
                spaceState = True
                splitableString += ","
            if writingStr[i] != " ":
                spaceState = False
                splitableString += writingStr[i]

        with open('pValues.csv', 'w') as f:
            f.write(splitableString)
        pvaluesData = pd.read_csv('pValues.csv')
        pvalues = pvaluesData['P>|t|'].reset_index()
        pvalues = pvalues.sort_values(by=['P>|t|'], ascending=False)
        for i in range(0,count):
            self.data = self.data.drop([self.tableHeaders[int(pvalues.iloc[i]['index'])]], axis=1)
        return self.data

    def setAlgorithms(self,type="Regresssion"):
        self.type = type
        if(type!="Classification"):
           self.algorithms =[
               RandomForestRegressor(),
               XGBRegressor(),
               LGBMRegressor(),
               GradientBoostingRegressor(),
               KNeighborsRegressor(),
               DecisionTreeRegressor(),
               CatBoostRegressor(),
               SVR(),
               XGBRFRegressor(),
               BaggingRegressor(),
               PolynomialFeatures(),
               LinearRegression(),
               PLSRegression(),
               Ridge(),
               Lasso(),
               ElasticNet(),
           ]
        else:
            self.algorithms = [
                RandomForestClassifier(),
                XGBClassifier(),
                LGBMClassifier(),
                GradientBoostingClassifier(),
                KNeighborsClassifier(),
                DecisionTreeClassifier(),
                CatBoostClassifier(),
                SVC(),
                LogisticRegression(),
                GaussianNB(),
            ]
    def getAlgorithms(self):
        return self.algorithms
    def getResults(self):
        self.names = []
        self.distances = []
        self.sqrts = []
        for i in range(0, len(self.algorithms)):
            name = str(self.algorithms[i]).split('(')[0]

            if(name.__contains__('CatBoostRegressor')):
                name = 'CatBoostRegressor'
            if (name.__contains__('CatBoostClassifier')):
                    name = 'CatBoostClassifier'
            if(name == 'PolynomialFeatures'):
                name = 'PolynomialRegressor'
            self.names.append(name)
            print(name)
            if(name != 'PolynomialRegressor'):
                self.algorithms[i].fit(self.x_train,self.y_train)
                y_pred = self.algorithms[i].predict(self.x_test)
            else:
                x_poly = self.algorithms[i].fit_transform(self.x_train)
                lin_reg = LinearRegression()
                lin_reg.fit(x_poly, self.y_train)
                y_pred = lin_reg.predict(self.algorithms[i].fit_transform(self.x_test))

            f = abs(self.y_test - y_pred).mean()
            self.distances.append(f)
            self.resultDf = pd.DataFrame(data=self.names,columns=['Algorithms'])
            if self.type!='Classification':
                self.resultDf['Distances'] = self.distances
                self.sqrts.append(np.sqrt(mean_squared_error(self.y_test, y_pred)))
            else:
                self.sqrts.append(accuracy_score(self.y_test, y_pred))
                self.confusionMatrixes.append(confusion_matrix(self.y_test,y_pred))
                self.f1Scores.append(f1_score(self.y_test,y_pred))
                self.macroAvgs.append(f1_score(self.y_test,y_pred,average='macro'))
                self.microAvgs.append(f1_score(self.y_test, y_pred, average='micro'))
                self.precisionScores.append(precision_score(self.y_test,y_pred))
                self.recallScores.append(recall_score(self.y_test,y_pred))

        if self.type == 'Classification':
            self.resultDf['F1 Scores'] = self.f1Scores;
            self.resultDf['Accuracy Scores'] = self.sqrts;
            self.resultDf['Precision Scores'] = self.precisionScores
            self.resultDf['Recall Scores'] = self.recallScores
            self.resultDf['Micro Averages'] = self.microAvgs
            self.resultDf['Macro Averages '] = self.macroAvgs
            self.resultDf['Confision Matrix '] = self.confusionMatrixes
        else:
            self.resultDf['Scores'] = self.sqrts
            self.resultDf = self.resultDf.sort_values(by=['Scores'])

        if(self.firstScoreTable.size == 0):
            self.firstScoreTable = self.resultDf
        return self.resultDf

    def getSortedResultDf(self,by,ascending):
        return self.resultDf.sort_values(by=[by],ascending=ascending)

    def randomForestCv(self):
        self.algorithms[0].random_state = 42
        rf_params = {'max_depth': list(range(1, 10)),
                     'max_features': [3, 5, 10, 15],
                     'n_estimators': [100, 200, 500, 1000, 2000]}
        rf_cv_model = GridSearchCV(self.algorithms[0],
                                   rf_params,
                                   cv=10,
                                   n_jobs=-1)
        rf_cv_model.fit(self.x_train, self.y_train)
        bestParams = rf_cv_model.best_params_
        if(str(self.algorithms[0]).split('(')[0]=="RandomForestRegressor"):
            self.algorithms[0] = RandomForestRegressor(max_depth=bestParams['max_depth'],max_features=bestParams['max_features'],n_estimators=bestParams['n_estimators'])
        else:
            self.algorithms[0] = RandomForestClassifier(max_depth=bestParams['max_depth'],max_features=bestParams['max_features'],n_estimators=bestParams['n_estimators'])

        return self.algorithms[0]


    def xgbCv(self):
        xgb_grid = {
            'colsample_bytree': [0.4, 0.5, 0.6, 0.9, 1],
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [2, 3, 4, 5, 6],
            'learning_rate': [0.1, 0.01, 0.5]
        }
        xgb_cv = GridSearchCV(self.algorithms[1],param_grid=xgb_grid,cv=10,n_jobs=-1,verbose=2)
        xgb_cv.fit(self.x_train, self.y_train)
        bestParams = xgb_cv.best_params_
        if(str(self.algorithms[1]).split('(')[0]=="XGBClassifier"):
            self.algorithms[1] = XGBClassifier(colsample_bytree=bestParams['colsample_bytree'],learning_rate=bestParams['learning_rate'], max_depth=bestParams['max_depth'],n_estimators=bestParams['n_estimators'])
        else:
            self.algorithms[1] = XGBRegressor(colsample_bytree=bestParams['colsample_bytree'],
                                               learning_rate=bestParams['learning_rate'],
                                               max_depth=bestParams['max_depth'],
                                               n_estimators=bestParams['n_estimators'])
        return self.algorithms[1]


    def lgbmCv(self):
        lgbm_grid = {
            'colsample_bytree': [0.4, 0.5, 0.6, 0.9, 1],
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'n_estimators': [20, 40, 100, 200, 500, 1000],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8]}
        lgbm_cv_model = GridSearchCV(self.algorithms[2], lgbm_grid, cv=10, n_jobs=-1, verbose=2)
        lgbm_cv_model.fit(self.x_train, self.y_train)
        bestParams = lgbm_cv_model.best_params_
        if (str(self.algorithms[2]).split('(')[0] == "LGBMRegressor"):
            self.algorithms[2] = LGBMRegressor(colsample_bytree=bestParams['colsample_bytree'],
                                                   learning_rate=bestParams['learning_rate'],
                                                   max_depth=bestParams['max_depth'],
                                                   n_estimators=bestParams['n_estimators'])
        else:
            self.algorithms[2] = LGBMClassifier(colsample_bytree=bestParams['colsample_bytree'],
                                               learning_rate=bestParams['learning_rate'],
                                               max_depth=bestParams['max_depth'],
                                               n_estimators=bestParams['n_estimators'])
        return self.algorithms[2]


    def gbmCV(self):
        gbm_params = {
            'learning_rate': [0.001, 0.01, 0.2],
            'max_depth': [5, 8,13,25, 50,75, 100],
            'n_estimators': [50,150,200,500,1000],
            'subsample': [0.25, 0.5, 0.75],
        }
        gbm_params['learning_rate'].append(self.algorithms[3].learning_rate)
        gbm_params['max_depth'].append(self.algorithms[3].max_depth)
        gbm_params['n_estimators'].append(self.algorithms[3].n_estimators)
        gbm_params['subsample'].append(self.algorithms[3].subsample)

        gbm_cv_model = GridSearchCV(self.algorithms[3], gbm_params, cv=10, n_jobs=-1, verbose=2)
        gbm_cv_model.fit(self.x_train, self.y_train)
        bestParams = gbm_cv_model.best_params_
        self.algorithms[3].learning_rate = bestParams['learning_rate']
        self.algorithms[3].max_depth = bestParams['max_depth']
        self.algorithms[3].n_estimators = bestParams['n_estimators']
        self.algorithms[3].subsample = bestParams['subsample']
        return self.algorithms[3]


    def knnCv(self):
        knn_params = {'n_neighbors': np.arange(1, 50)}
        knn_cv_model = GridSearchCV(self.algorithms[4], knn_params, cv=10)
        knn_cv_model.fit(self.x_train, self.y_train)
        print(knn_cv_model.best_params_)
        if (str(self.algorithms[4]).split('(')[0] == "KNeighborsRegressor"):
            self.algorithms[4] = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
        else:
            self.algorithms[4] = KNeighborsClassifier(n_neighbors=knn_cv_model.best_params_["n_neighbors"])

        return self.algorithms[4]


    def decisionTreeCv(self):
        cart_grid = {"max_depth": range(1, 10),
                     "min_samples_split": list(range(2, 50))}
        cart_cv = GridSearchCV(self.algorithms[5], cart_grid, cv=10, n_jobs=-1, verbose=2)
        cart_cv_model = cart_cv.fit(self.x_train, self.y_train)
        bestParams = cart_cv_model.best_params_
        if (str(self.algorithms[5]).split('(')[0] == "DecisionTreeClassifier"):
            self.algorithms[5] = DecisionTreeClassifier(max_depth=bestParams['max_depth'], min_samples_split=bestParams['min_samples_split'])
        else:
            self.algorithms[5] = DecisionTreeRegressor(max_depth=bestParams['max_depth'],
                                                        min_samples_split=bestParams['min_samples_split'])
        return self.algorithms[5]


    def catBoostCv(self):
        catb_params = {
            'iterations': [200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth'         : [4,5,6,7,8,9, 10]}
        catb_cv_model = GridSearchCV(self.algorithms[6], catb_params, cv=10, n_jobs=-1, verbose=2)
        catb_cv_model.fit(self.x_train, self.y_train)
        bestParams = catb_cv_model.best_params_
        print(bestParams)
        if self.type == 'Classification':
            self.algorithms[6] = CatBoostClassifier(iterations = bestParams['iterations'],
                                                    learning_rate = bestParams['learning_rate'],
                                                    depth = bestParams['depth'])
        else:
            self.algorithms[6] = CatBoostRegressor(iterations = bestParams['iterations'],
                                                    learning_rate = bestParams['learning_rate'],
                                                    depth = bestParams['depth'])
        return self.algorithms[6]


    def supportVectorCv(self):
        svc_params = {"C": np.arange(1, 200)}
        svc_cv_model = GridSearchCV(self.algorithms[7], svc_params,
                                    cv=10,
                                    n_jobs=-1,
                                    verbose=2)
        svc_cv_model.fit(self.x_train, self.y_train)
        bestParams = svc_cv_model.best_params_
        if (self.type=='Classification'):
            self.algorithms[7] = SVC(kernel = 'linear', C = bestParams['C'])
        else:
            self.algorithms[7] = SVR(kernel='linear', C=bestParams['C'])


    def ridgeCv(self):
        lambdalar = 10 ** np.linspace(10, -2, 100) * 0.5
        self.algorithms[13] = RidgeCV(alphas=lambdalar,
                           scoring="neg_mean_squared_error",
                           normalize=True)
        return self.algorithms[13]


    def baggedTreesCV(self):
        bag_params = bag_params = {"n_estimators": range(2,20)}
        bag_cv_model = GridSearchCV(self.algorithms[9], bag_params, cv=10)
        bag_cv_model.fit(self.x_train, self.y_train)
        bestParams = bag_cv_model.best_params_
        self.algorithms[9] = BaggingRegressor(n_estimators=bestParams['n_estimators'], random_state=1)
        return self.algorithms[9]


    def lassoCv(self):
        lasso_cv_model = LassoCV(alphas=None,
                                 cv=10,
                                 max_iter=10000,
                                 normalize=True)
        lasso_cv_model.fit(self.x_train, self.y_train)
        self.algorithms[14]  = Lasso(alpha = lasso_cv_model.alpha_)
        return self.algorithms[14]

    def elasticNetCv(self):
        enet_cv_model = ElasticNetCV(cv=10, random_state=0).fit(self.x_train, self.y_train)
        self.algorithms[15] = ElasticNet(alpha=enet_cv_model.alpha_)
        return self.algorithms[15]

    def allCv(self):
        print(str(datetime.datetime.now()).split('.')[0])
        self.randomForestCv()
        self.xgbCv()
        self.lgbmCv()
        self.gbmCV()
        self.knnCv()
        self.decisionTreeCv()
        self.catBoostCv()
        self.supportVectorCv()
        print(str(datetime.datetime.now()).split('.')[0])