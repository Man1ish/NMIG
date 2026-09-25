import pandas as pd
import time
import numpy as np
import json
"""
This class provides the dataset of invocation and functions
"""
class DatasetLoader:

    def __init__(self, **kwargs):
        # self.data_path = kwargs.get('data')
        # self.threshold = kwargs.get('threshold', 14400)  # default if not passed
        # self.top_functions_no = kwargs.get('top_functions_no', 7)
        # self.function_path = kwargs.get('function_path')
        # self.function_pattern = kwargs.get('function_pattern', 'normal')
        # self.hour_trace = kwargs.get('hour_trace', 4) # this is for how many hours the trace is
        # self.trace_id = kwargs.get('trace_id', 7) # Each trace is calculated by self.hour_trace * 60 * 60
        # self.filtered_df = self.load()
        # self.functions = self.load_functions()
        self.dataset = kwargs.get('dataset')
        self.filtered_df = self.load()
        self.functions = self.load_functions()
        self.function_policy = self.load_profile_policy()
        
        
        
    def load_profile_policy(self):
        # read json file and convert to dict
        data = {}
        with open('dataset/latency_first.json') as f:
            data = json.load(f)
        return data

    def load(self):
        # Load the dataset
        if self.dataset == "normal":
            df = pd.read_csv('dataset/normal_df.csv')
            self.function_path = 'dataset/normal_functions.csv'
        elif self.dataset == "brusty":
            df = pd.read_csv('dataset/brusty_df.csv')
            self.function_path = 'dataset/brusty_functions.csv'
        elif self.dataset == "overall":
            df = pd.read_csv('dataset/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt')
            self.function_path = 'dataset/functions.csv'

            df['arrival_time'] = df['end_timestamp'] - df['duration']
            df.sort_values(by=['arrival_time'], inplace=True, ascending=True)
            df['arrival_time_fix'] = df['arrival_time'] - df['arrival_time'].iloc[0]
            df.sort_values(by=['arrival_time_fix'], inplace=True, ascending=True)
        elif self.dataset == "similar":
            # df = pd.read_csv('dataset/similar_df.csv')
            # self.function_path = 'dataset/similar_functions.csv'
            df = pd.read_csv('dataset/similar_gen_df.csv')
            self.function_path = 'dataset/similar_gen_functions.csv'
        elif self.dataset == "similar_gen":
            df = pd.read_csv('dataset/similar_gen_df.csv')
            self.function_path = 'dataset/similar_gen_functions.csv'
        elif self.dataset == "cold_gen":
            df = pd.read_csv('dataset/cold_gen_df.csv')
            self.function_path = 'dataset/cold_gen_functions.csv'
        elif self.dataset == "bursty_gen":
            df = pd.read_csv('dataset/bursty_gen_df.csv')
            self.function_path = 'dataset/bursty_gen_functions.csv'
        elif self.dataset == "normal_gen":
            df = pd.read_csv('dataset/normal_gen_df.csv')
            self.function_path = 'dataset/normal_gen_functions.csv'
        elif self.dataset == "hybrid_gen":
            df = pd.read_csv('dataset/hybrid_gen_df.csv')
            self.function_path = 'dataset/hybrid_gen_functions.csv'
        elif self.dataset == "control_trace":
            df = pd.read_csv('dataset/controlled_1hour_500_requests.csv')
            self.function_path = 'dataset/controlled_1hour_500_requests_functions.csv'
        else:
            raise ValueError("Invalid dataset type. Choose 'normal' or 'brusty'.")
        
        # Sort the dataframe by arrival_time
        df.sort_values(by='arrival_time', ascending=True)
        
        return df
    
    def load_functions(self):
        data =  pd.read_csv(self.function_path)
        unique_functions = self.filtered_df['func'].unique()
        
        # from data extract the unique functions rows and return them
        data = data[data['id'].isin(unique_functions)]
        df_dict = data.set_index('id').to_dict(orient='index')
        return df_dict
