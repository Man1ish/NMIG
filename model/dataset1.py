import pandas as pd
import time
import numpy as np
class Dataset:

    def __init__(self, **kwargs):
        self.data_path = kwargs.get('data')
        self.threshold = kwargs.get('threshold', 14400)  # default if not passed
        self.top_functions_no = kwargs.get('top_functions_no', 7)
        self.function_path = kwargs.get('function_path')
        self.function_pattern = kwargs.get('function_pattern', 'normal')
        self.hour_trace = kwargs.get('hour_trace', 4) # this is for how many hours the trace is
        self.trace_id = kwargs.get('trace_id', 7) # Each trace is calculated by self.hour_trace * 60 * 60
        self.filtered_df = self.load()
        self.functions = self.load_functions()
        


    def load(self):
        df = pd.read_csv(self.data_path)

        df['arrival_time'] = df['end_timestamp'] - df['duration']
        df.sort_values(by=['arrival_time'], inplace=True, ascending=True)
        df['arrival_time_fix'] = df['arrival_time'] - df['arrival_time'].iloc[0]
        df.sort_values(by=['arrival_time_fix'], inplace=True, ascending=True)
        # df = df[df['arrival_time'] <= self.threshold]

        # Step 1: Assign 4-hour trace windows
        df['trace_id'] = (df['arrival_time'] // (self.hour_trace * 60 * 60)).astype(int)

        # Step 2: Create 1-minute bins
        df['minute_bin'] = (df['arrival_time'] // 60).astype(int)


        # Step 2.1 : Filter out the dataframe based on trace_id
        df = df[df['trace_id'] == self.trace_id]

        # Step 3: Count number of requests per (function, minute_bin)
        requests_per_minute = df.groupby(['func', 'minute_bin']).size().reset_index(name='request_count')

        # Step 4: Pivot to get time series (rows = minute_bin, columns = func, values = request_count)
        pivot_df = requests_per_minute.pivot_table(index='minute_bin', columns='func', values='request_count', fill_value=0)

        # Step 5: Calculate Coefficient of Variation (CoV) for each function
        # CoV = Standard Deviation / Mean
        cov_per_function = pivot_df.std(ddof=0) / pivot_df.mean()

        total_requests_per_function = df.groupby('func').size()
        


        # Step 5: Combine both into a DataFrame
        cov_df = pd.DataFrame({
            'func': cov_per_function.index,
            'CoV': cov_per_function.values,
            'total_requests': total_requests_per_function[cov_per_function.index].values
        })

        # Step 6: Sort by highest CoV
        # cov_df = cov_df.sort_values(by='total_requests', ascending=False)
        cov_df = cov_df.sort_values(by='total_requests', ascending=False)


        print(cov_df.head(40))
        exit()



       
        # Step 3: Count invocations per minute per trace_id and func
        # grouped = df.groupby(['trace_id', 'func', 'minute_bin']).size().reset_index(name='invocation_count')

        # # Step 4: Ensure all 240 minutes are present for each trace_id and func
        # filled_counts = []

        # for (trace_id, func), group in grouped.groupby(['trace_id', 'func']):
        #     minute_range = range(trace_id * 240, (trace_id + 1) * 240)
        #     all_minutes = pd.DataFrame({'minute_bin': minute_range})
        #     all_minutes['trace_id'] = trace_id
        #     all_minutes['func'] = func
        #     merged = all_minutes.merge(group, on=['trace_id', 'func', 'minute_bin'], how='left').fillna(0)
        #     filled_counts.append(merged)

        # counts_filled = pd.concat(filled_counts, ignore_index=True)

        # cov_df = counts_filled.groupby(['trace_id', 'func']).apply(self.compute_cov).reset_index()
        # cov_df['pattern'] = cov_df['cov'].apply(self.classify_pattern)

        # # Count how many times each pattern appears
        # pattern_counts_by_trace = cov_df.groupby(['trace_id', 'pattern']).size().reset_index(name='pattern_count')


        # # # # Merge back into cov_df
        # cov_df = cov_df.merge(pattern_counts_by_trace, on=['trace_id', 'pattern'], how='left')

        # top_func_names = df[df['trace_id'] == self.trace_id]['func'].value_counts().head(self.top_functions_no).index.tolist()

        # filtered_df = df[(df['trace_id'] == self.trace_id) & (df['func'].isin(top_func_names))]
        # # Lookup with tuple values
        # lookup = {row['func']: (row['pattern'], row['cov'], row['pattern_count']) for _, row in cov_df.iterrows()}

        # # Correct way to access tuple elements by index
        # filtered_df['pattern'] = filtered_df['func'].map(lambda f: lookup.get(f, (None, None))[0])
        # filtered_df['cov'] = filtered_df['func'].map(lambda f: lookup.get(f, (None, None))[1])
        # filtered_df['count'] = filtered_df['func'].map(lambda f: lookup.get(f, (None, None))[2])
        # filtered_df = filtered_df[filtered_df['pattern'] == self.function_pattern]

        # filtered_df = filtered_df.sort_values(by='arrival_time').reset_index(drop=True)


        print(filtered_df)
        exit()
        exit()

        # cov_df where pattern is self.function_pattern
        # if self.function_pattern == 'normal':
        #     cov_df = cov_df[cov_df['pattern'] == 'normal']
        # else:
        #     cov_df = cov_df[cov_df['pattern'] == self.function_pattern]
        
        # top_functions = cov_df['func'].value_counts().head(self.top_functions_no).index



        # filtered_df = df[df['func'].isin(top_functions)]
        # filtered_df = filtered_df.sort_values(by='arrival_time')

        # print(filtered_df)
        # exit()

        return filtered_df
    
    # Step 5: Compute CoV for each trace_id and func
    def compute_cov(self, group):
        mean = group['invocation_count'].mean()
        std = group['invocation_count'].std()
        cov = std / mean if mean > 0 else np.nan
        return pd.Series({'mean': mean, 'std': std, 'cov': cov})

    

    # Step 6: Classify patterns
    def classify_pattern(self, cov):
        if pd.isna(cov):
            return 'unknown'
        elif cov < 1:
            return 'predictable'
        elif cov < 4:
            return 'normal'
        else:
            return 'bursty'
    

    
    def load_functions(self):
        data =  pd.read_csv(self.function_path)
        unique_functions = self.filtered_df['func'].unique()
        # from data extract the unique functions rows and return them
        data = data[data['id'].isin(unique_functions)]
        df_dict = data.set_index('id').to_dict(orient='index')

        return df_dict

    def simulate(self):
        # Get unique arrival times from the filtered dataframe
        unique_arrival_times = self.filtered_df['arrival_time'].unique()
        start_time = time.time()
        previous_time = 0  # Track previous arrival time

        # Handle events at 0 immediately
        events_at_zero = self.filtered_df[self.filtered_df['arrival_time'] == 0]
        for _, row in events_at_zero.iterrows():
            print(f"Function: {row['func']} - Arrival Time: {row['arrival_time']} seconds")

        threshold_second = 1050
        # Process the rest
        for arrival_time in unique_arrival_times:
            if arrival_time == 0:
                continue

            # Calculate the duration to sleep based on the difference between consecutive events
            sleep_time = arrival_time - previous_time
            previous_time = arrival_time  # Update previous time to current event's time

            # Check if the total elapsed time exceeds 2 minutes
            
            if time.time() - start_time > threshold_second:
                print("Breaking after 2 minutes.")
                break

            # Sleep the calculated duration
            time.sleep(sleep_time)

            # Print events scheduled for the current arrival time
            events_at_time = self.filtered_df[self.filtered_df['arrival_time'] == arrival_time]
            for _, row in events_at_time.iterrows():
                print(f"Function: {row['func']} - Arrival Time: {row['arrival_time']} seconds")

        
