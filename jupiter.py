import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

class Moons:
	def __init__(self,db_path):
		self.data = pd.DataFrame()
		self.load_data(db_path)
		self.X_test = None
		self.y_test = None

	def load_data(self,db_path):
		connect = sqlite3.connect(db_path)
		query = "SELECT * FROM Moons"
		self.data = pd.read_sql_query(query, connect)
		#makes a pandas Dataframe above
		connect.close()

	def basic_dataset_info(self):
		print (self.data.info())

	def sum_stat(self):
		#summary stats for all numerical columns
		return self.data.describe()

	def num_null_values(self):
		print ("The number of null values for each columns are")
		return self.data.isnull().sum()

	def calculate_missing_percentage(self):
		missing_percentage = self.data.isnull().mean()*100
		return missing_percentage

	def drop_missing_values(self, axis=0, inplace=False):
		#default is to drop rows but we can chooose to drop columns
		return self.data.dropna(axis=axis, inplace=inplace)

	def visualise_missing_data(self):
		plt.figure(figsize=(10,8))
		sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
		plt.title('Missing Data : Yellow means missing values')
		plt.show()

	def visualise_corr(self):
		#visualising correlations between a lot of num variables
		corr_matrix = self.data.corr()

		plt.figure(figsize=(10,8))

		sns.heatmap(corr_matrix, annot=True, cmap="crest", vmin = -1 , vmax=1)

		plt.title("Correlation Matrix")
		plt.xlabel("X - Features")
		plt.ylabel("Features")
		plt.show()

	#CATEGORY METHODS
	#creating multiple categorical plots
	def create_mult_catplots(self,x,y_list,kind,height=4,aspect=1.2):
		for y_var in y_list:
			sns.catplot(x=x, y=y_var, kind=kind, data=self.data, height=height, aspect=aspect)
			plt.show()

	#NUMERICAL METHODS
	#Creating pair_plots for the numerical features
	#Summing large amounts of data in a single figure
	#to check if there's noticeable patterns present
	def pair_plots(self,attributes, hue=None):
		if hue:
			sns.pairplot(self.data[attributes], hue="group")
			plt.show()
		else:
			sns.pairplot(self.data[attributes])
			plt.show()

	#Creating histogram plot for numerical features
	#Represents distribution of many variables
	def create_hist_plots(self, numerical_features):
		for feature in numerical_features:
			plt.figure(figsize=(8, 6))
			sns.histplot(self.data[feature], kde=True)
			plt.title(f"Distribution of {feature}")
			plt.show()

	#ANALYSING DISTRIBUTION METHODS
	#Method to Create relational plot (scatter plot)
	def create_relplot(self,x_data,y_data):
		#creating a seaborn relplot
		sns.relplot(x=x_data,y=y_data, data = self.data,hue="group")
		plt.title(f"Scatter Plot of {x_data} by {y_data}")
		plt.show()

	#Method to create line plots
	def create_line_plot(self,x_data,y_data):
		plt.figure(figsize=(12,8))
		sns.set_theme(style="darkgrid")
		sns.lineplot(x=x_data,y=y_data,data=self.data)
		plt.title(f"Time Series Plot of {x_data} by {y_data}")
		plt.show()

	#Extracting particular moon entries and reusing it as a Pandas DataFrame
	def extract_moon_data(self,moon_names,db_path):
		moon_data = self.data[self.data['moon'].isin(moon_names)]
		print(moon_data)
		#Create new instance of the class with the extracted data
		new_instance = Moons(db_path=db_path)
		new_instance.data = moon_data

		return new_instance

	#Regression Model to predict values
	def linear_regression_model(self,test_size):
		#Convertions
		self.data['period_seconds'] = self.data['period_days'] *24 *60*60
		self.data['distance_metres'] = self.data['distance_km'] * 1000

		#Calculate the x-values, y-values
		self.data['T_squared'] = self.data['period_seconds']**2
		self.data['a_cubed'] = self.data['distance_metres']**3

		#Give clear variable names
		X = self.data[['T_squared']]
		y= self.data['a_cubed']

		#We split the data for training and testing.
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)

		#adding test data as an attribute
		self.X_test, self.y_test = X_test, y_test

		#Linear Regression Model
		model = LinearRegression()

		#Train the Model
		model.fit(X_train, y_train)
		#Testing the model
		y_pred = model.predict(X_test)

		#Values to determine accuracy of the model
		mse = mean_squared_error(y_test,y_pred)
		r2 = r2_score(y_test,y_pred)

		print(f"The Mean Squared Error is {mse}")
		print(f"The R-square is {r2}")

		#Model Parameters
		print(f"Line gradient from model: {model.coef_[0]}")
		print(f"Line intercept from model: {model.intercept_}")

		#Labels given for the graph
		fig, ax =plt.subplots()
		ax.scatter(X_test,y_test, color='blue', label="Actual data")
		ax.plot(X_test, y_pred, color="black", label="Predicted data")
		plt.title('Linear Regression Model')
		ax.set_xlabel('T^2')
		ax.set_ylabel('a^3')
		ax.legend()

		#here, we return the model
		return model

	#checking difference between predicted and actual values
	def plot_residuals(self,model):
		y_pred = model.predict(self.X_test)
		residuals = self.y_test - y_pred

		#plotting the graph
		plt.scatter(self.X_test,residuals,color="orange",alpha=0.5)
		plt.axhline(y=0,color="black", linestyle='--')
		plt.title("Residuals Plot")
		plt.xlabel("Predicted Values")
		plt.ylabel("Residuals")
		plt.show()

	#function to estimate the mass of Jupiter
	def estimate_mass(self,model):
		G_known = 6.67e-11
		gradient = model.coef_[0]

		#Using kepler's third law
		mass_estimate = (4 * np.pi**2 * gradient) / G_known 

		real_mass_jupiter = 1.898e27 #this is in kg
		relative_error = ((mass_estimate-real_mass_jupiter)/real_mass_jupiter) * 100

		print(f"Estimated Mass of Jupiter:{mass_estimate}kg")
		print(f"Relative Error: {relative_error}%")



