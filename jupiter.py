#jupiter.py
import sqlite3 #
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
		self.dataset = pd.read_sql_query(query, connect)
		connect.close()

	def basic_dataset_info(self):
		print (self.dataset.info())

	def sum_stat(self):
		#summary stats for all numerical columns 
		return self.dataset.describe()

	def num_null_values(self):
		print ("The number of null values for each columns are")
		return self.dataset.isnull().sum()

	def visualise_corr(self):
		#visualising correlations between all num variables
		corr_matrix = self.dataset.corr()

		plt.figure(figsize=(10,8))

		sns.heatmap(corr_matrix, annot=True, cmap="crest", vmin = -1 , vmax=1)

		plt.title("Correlation Matrix")
		plt.show()

	def corr_group_and_numerical_factors(self,x-value):
		#plotting distribution of distances from moon to jupiter
		sns.displot(data = self.data, x="x-value", col="group")

	def extract_moon_data(self,moon_name):
		moon_data = self.dataset[self.dataset['moon'] == moon_name]
		return moon_data

