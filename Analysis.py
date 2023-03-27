from sklearn.model_selection import train_test_split
import pandas as pd
from Graphics import Graphics  # import class of method

# set_options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class Analysis(Graphics):
    def __init__(self, df: pd.DataFrame, show_info: bool = True):
        super.__init__()
        self.df = df
        self.show_info = show_info
    def run(self) -> dict:
        self.printer('data split into train nd test')

        train, test = self.splitTrainTest()
        print(f'train shape: {train.shape}\n')
        print(f'test shape: {test.shape}\n')

        self.printer('Visualitation of data')

        if self.show_info:
            print("\n  Dtypes \n")
            print(train.dtypes)
            print("\n  Head \n")
            print(train.head().T)
            print("\n  Describe \n")
            print(train.describe().T)

        self.printer('A heatmap respect to price ')
        self.heatmap(train,"01-trainHeatmap.png")

