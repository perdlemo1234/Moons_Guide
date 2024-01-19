import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Moons:
	def __init__(self,db_path):
		self.data = None
		self.load_data(db_path)

	def load_data(self,db_path):
		connect = sqlite3.connect(db_path)
		query = "SELECT * FROM Moons"
		self.data = pd.read_sql_query(query, connect)
		connect.close()

	def basic_dataset_info(self):
		print (self.data.info())

	def sum_stat(self):
		#summary stats for all numerical columns 
		return self.data.describe()

	def num_null_values(self):
		print ("The number of null values for each columns are")
		return self.data.isnull().sum()

	def visualise_corr(self):
		#visualising correlations between all num variables
		corr_matrix = self.data.corr()

		plt.figure(figsize=(10,8))

		sns.heatmap(corr_matrix, annot=True, cmap="crest", vmin = -1 , vmax=1)

		plt.title("Correlation Matrix")
		plt.xlabel("X - Features")
		plt/ylabel("Features")
		plt.show()

	#CATEGORY METHODS
	#creating multiple categorical plots
	def create_mult_catplots(self,x,y_list,kind,height=4,aspect=1.2):
		for y_var in y_list:
			sns.catplot(x=x, y=y_var, kind=kind, data=self.data, height=height, aspect=aspect)
			plt.show()

	#Creating count_plots for categorical features
	def create_count_plots(self, categorical_features)
		for feature in  categorical_features:
			plt.figure(figsize=(8,6))
			sns.countplot(x=feature, data=self.data)
			plt.title(f"Count Plot of {feature})
			plt.show()


	#NUMERICAL METHODS
	#Creating pair_plots for the numerical features
	def pair_plots(self,attributes,hue=None):
		if hue:
			sns.pairplot(self.data[attributes], hue=hue)
		else:
			sns.pairplot(self.data[attributes])
			plt.show()

	#Creating histogram plot for numerical features
	def create_distribution_plots(self, numerical_features):
		for feature in numerical_features:
			plt.figure(figsize=(8, 6))
			sns.histplot(self.data[feature], kde=True)
			plt.title(f'Distribution of {feature}')
			plt.show()

	#ANALYSING DISTRIBUTION METHODS
	#Method to Create relational plot (scatter plot)
	def create_relplot(self,x_data,y_data):
		#creating a seaborn relplot
		sns.relplot(x=x_data,y=y_data, data = self.data)
		plt.show()

	#Method to create violin plot to explore distribution of data
	def create_violin_plots(self,x_data,y_data):
		plt.figure(figsize=(10,6))
		sns.violinplot(x=x_data,y=y_data, data=self.data)
		plt.title(f"Violing Plot of {x} by {y}")
		plt.show()

	#Method to create line plots
	def create_line_plot(self,x_data,y_data):
		plt.figure(figsize=(12,8))
		sns.lineplot(x=x_data,y=y_data,data=self.data)
		plt.title(f"Time Series Plot of {y} over {x}")
		plt.show()




	def extract_moon_data(self,moon_names):
                moon_data = self.data[self.data['moon'].isin(moon_names)]
                return moon_data


