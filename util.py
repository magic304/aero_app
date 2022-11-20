
# 根据file路径将csv转化为dataframe
import pandas as pd


def found_df(file_path):
    df_datas = None
    if file_path.endswith('.csv'):
        df_datas = pd.read_csv(file_path)
        # self.column_headers = list(self.df_datas.columns.values)
    elif file_path.endswith('.xlsx' or 'xls'):
        df_datas = pd.read_excel(file_path, engine='openpyxl')
        # self.column_headers = list(self.df_datas.columns.values)
    else:
        print("文件类型错误，请插入csv或xlsx类型的文件")
    return df_datas