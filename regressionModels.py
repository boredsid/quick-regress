import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class DataModel:

	def __init__(self, file):
		self.df = pd.read_csv(file)
		self.column_list = list(self.df.columns)

	def outputColumnSelector(self, y_col):
		self.yColumn = y_col
		self.yColumnIndex = self.column_list.index(self.yColumn)

	def simpleRun(self):
		self.inputOutput()
		self.splitData()
		self.fitModels()
		self.modelTest()
		self.bestModel = max(self.r2_scores, key = self.r2_scores.get)

	def inputOutput(self):
		self.X = self.df.iloc[:,self.df.columns!=self.yColumn].values
		self.y = self.df.iloc[:,self.yColumnIndex].values

	def splitData(self, test_size=0.2):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)

	def fitModels(self, poly_degree=10, svr_kernel='rbf', rf_estimators=10):
		self.pf = PolynomialFeatures(degree=poly_degree)
		self.X_train_poly = self.pf.fit_transform(self.X_train)

		self.sc_X = StandardScaler()
		self.X_train_norm = self.sc_X.fit_transform(self.X_train)

		self.sc_y = StandardScaler()
		self.y_train_norm = self.sc_y.fit_transform(self.y_train.reshape(len(self.y_train),1))

		self.regressor_simple = LinearRegression()
		self.regressor_simple.fit(self.X_train, self.y_train)

		self.regressor_poly = LinearRegression()
		self.regressor_poly.fit(self.X_train_poly,self.y_train)

		self.regressor_svr = SVR(kernel = svr_kernel)
		self.regressor_svr.fit(self.X_train_norm,self.y_train_norm.ravel())

		self.regressor_tree = DecisionTreeRegressor()
		self.regressor_tree.fit(self.X_train,self.y_train)

		self.regressor_forest = RandomForestRegressor(n_estimators=rf_estimators)
		self.regressor_forest.fit(self.X_train,self.y_train)

	def modelTest(self):
		y_pred_simple = self.regressor_simple.predict(self.X_test)
		y_pred_poly = self.regressor_poly.predict(self.pf.transform(self.X_test))
		y_pred_svr_norm = self.regressor_svr.predict(self.sc_X.transform(self.X_test))
		y_pred_svr = self.sc_y.inverse_transform(y_pred_svr_norm.reshape(len(y_pred_svr_norm),1)).ravel()
		y_pred_tree = self.regressor_tree.predict(self.X_test)
		y_pred_forest = self.regressor_forest.predict(self.X_test)

		self.y_pred_test = {'simple':y_pred_simple,'poly':y_pred_poly,'svr':y_pred_svr,'tree':y_pred_tree,'forest':y_pred_forest}

		r2_simple = r2_score(self.y_test,y_pred_simple)
		r2_poly = r2_score(self.y_test,y_pred_poly)
		r2_svr = r2_score(self.y_test,y_pred_svr)
		r2_tree = r2_score(self.y_test,y_pred_tree)
		r2_forest = r2_score(self.y_test,y_pred_forest)

		self.r2_scores = {'simple':r2_simple,'poly':r2_poly,'svr':r2_svr,'tree':r2_tree,'forest':r2_forest}

	def modelPredict(self, X_pred):
		y_pred_simple = self.regressor_simple.predict(X_pred)
		y_pred_poly = self.regressor_poly.predict(self.pf.transform(X_pred))
		y_pred_svr_norm = self.regressor_svr.predict(self.sc_X.transform(X_pred))
		y_pred_svr = self.sc_y.inverse_transform(y_pred_svr_norm.reshape(len(y_pred_svr_norm),1)).ravel()
		y_pred_tree = self.regressor_tree.predict(X_pred)
		y_pred_forest = self.regressor_forest.predict(X_pred)