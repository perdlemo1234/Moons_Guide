import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Moons:
	def __init__(self,db_path):
		self.data = pd.DataFrame()
		self.load_data(db_path)

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

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skllearn.metrics import mean_squared _error
import matplotlib.pyplot as plt
import numpy as np

	def linear_regression_model(self,test_size):
		#Calculate the x-values, y-values
		self.data['T-squared'] = self.data['period_days']**2
		self.data['a_cubed'] = self.data['distance_km']**3

		#Give clear variable names to the values
		X = self.data[['T_squared']]
		y= self.data['a_cubed']

		#We split the data for training and testing.
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)

		model = LinearRegression()

		#Train the Model
		model.fit(X_train, y_train)
		#Testing the model
		y_pred = model.predict(X_test)

		#mean-squared-error for the model - how accurate is it?
		mse = mean_squared_erro(y_test,y_pred)

		#Labels given for the graph
		plt.scatter(X_test,y_test, color='blue')
		plt.plot(X_test, y_pred, color="black")
		plt.title('Linear Regression Model')
		plt.xlabel('T^2')
		plt.ylabel('a^3')
		plt.show()

		#here, we return the model 
		return model

	def estimate_mass(self,model):
		G_known = 6.67e-11
		gradient = model.coef_[0]
		mass_estimate = (4 * np.pi**2 * gradient) / G_known 

		print(f"Estimated Mass of Jupiter:{mass_estimate}kg")

