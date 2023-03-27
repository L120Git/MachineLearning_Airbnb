import math 
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import pandas as pd
import numpy as np

class Graphics: 
    def __init__(self, path: str = "./imgs/"):
        self.path = path #initialize the directory
    
    def createHeatmap(self, df: pd.DataFrame, target_name: str, file_name: str) -> None:
        '''
        Create and save an image with correlation-heatmap without target column
        :param df: DataFrame
        :param target_name: name of target column
        :param file_name: name and file extension
        :return: None
        '''
        if not os.path.exists(f'{self.path}{file_name}'):
            corr = np.abs(df.drop([target_name], axis=1).corr())
            
            mask = np.zero_like(corr)
            mask[np.tribu_indices_from(mask)] = True
            
            plt.subplots(figsize=(12,10))
            
            sns.savefig(f'{self.path}{file_name}', bbox_inches='tight') #save the heatmap with size tight
            print(f'{self.path}{file_name} has been succesfully created')
        else: 
            print(f'{self.path}{file_name} alredy exist')
            
    def createHistogram(self, df: pd.Dataframe, file_name:str, object_type) -> None:
        '''
        Create and save a file png of histograms (numeric variables)
        :param object_type: object type of df data
        :param df: Dataframe
        :param file_name: name and file extension
        :return: None        
        '''
        
        if not os.path.exists(f'{self.path}{file_name}'):
            col_list =[z for z in df.columns if type(df[z].dtype) != object_type]
            counter_first = int(len(col_list)/4)+1
            counter_second = 4
            counter_third = 0
            
            for z in col_list:
                counter_third += 1 
                
                value_number = f'{df[z].count()}/{df.shape[0]}'
                plt.subplot(counter_first, counter_second, counter_third)
                df[z].plot.hist(alpha= 0.5, bins=25, grid=True)
                plt.xlabel(f'{z}{value_number}')
                plt.tight_layout()
            
            plt.savefig(f'{self.path}{file_name}')
            print(f'{self.path}{file_name} hass been succesfully created')
            
        else:
            print(f'{self.path}{file_name} already exist')
            
    def  createScatterMatrix(self, df:pd.DataFrame, file_name:str) -> None:
        '''
        Create and save a scatter_matrix as png file
        :param df: DataFrame
        :param file_name: name and file extension
        :return: None
        '''
        
        if not os.path.exists(f'{self.path}{file_name}'):
            pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(30,30), diagonal='hist')
            plt.savefig(f'{self.path}{file_name}')
            print(f'{self.path}{file_name} hass been succesfully created')
        else:
            print(f'{self.path}{file_name} alredy exist')
            
    def plotAlphaValues(self, apha_vector, scores, file_name):
        if not os.path.exist(f'{self.path}{file_name}'):
            fig, ax = plt.subplots()
            plt.xlabel ('alpha', frontsize=16)
            plt.ylabel('5-fold RMSE')
            #plt.ylim((0,1))
            plt.savefig(f'{self.path}{file_name}')
            print(f'{self.path}{file_name} hass been succesfully created')
        else:
            print(f'{self.path}{file_name} alredy exist')
            
            
    @staticmethod      
    def  plotFeaturesRank(X, f_test, feature_names, mi):
        plt.figure(figsize=(20,5))
        plt.subplot(1,2,1)
        plt.bar(range(X.shape[1]), f_test, align="center")
        plt.xticks(range(X.shape[1]), feature_names, rotation=90)
        plt.xlabel('features')
        plt.ylabel('Ranking')
        plt.title('F-info score')

        plt.subplot(1, 2, 2)
        plt.bar(range(X.shape[1]), mi, align="center")
        plt.xticks(range(X.shape[1]), feature_names, rotation=90)
        plt.xlabel('features')
        plt.ylabel('Ranking')
        plt.title('Mutual information score')

        plt.show()
    
    @staticmethod
    def Printer(text_to_print:str):
        
        print(f'\n#######################################################################################\n'
              f'----    {text_to_print}\n'
              f'#######################################################################################\n')
