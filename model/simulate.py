import time

class ExecutionStrategy:
    def execute(self, simulator):
        raise NotImplementedError("Each strategy must implement the 'execute' method.")

# Execute each function one time
class OneTimeExecution(ExecutionStrategy):
    def execute(self, simulator):
        # get unique func rows
        filtered_df = simulator.filtered_df
    
        # Get the first row for each unique category
        unique_df = filtered_df.drop_duplicates(subset="func", keep="first")
        for _, row in unique_df.iterrows():
            simulator.executor.handle_event(row)


class BrustyExecution(ExecutionStrategy):
    def execute(self, simulator):
        filtered_df = simulator.filtered_df
        executor = simulator.executor
        threshold_stop = simulator.threshold_stop
        

        unique_arrival_times = sorted(filtered_df['arrival_time'].unique())
        start_time = time.time()
        previous_time = 0

       
        events_at_zero = filtered_df[filtered_df['arrival_time'] == 0]
        for _, row in events_at_zero.iterrows():
            executor.handle_event(row)

        for arrival_time in unique_arrival_times:
            if arrival_time == 0:
                continue

            sleep_time = arrival_time - previous_time
            previous_time = arrival_time

            time.sleep(sleep_time)

            if time.time() - start_time > threshold_stop:
                print(f"Breaking after {threshold_stop} seconds (before executing {arrival_time}s events).")
                break

            events_at_time = filtered_df[filtered_df['arrival_time'] == arrival_time]
            for _, row in events_at_time.iterrows():
                executor.handle_event(row)

class CustomExecution(ExecutionStrategy):
    def execute(self, simulator):
        filtered_df = simulator.filtered_df
        executor = simulator.executor
        threshold_stop = simulator.threshold_stop
        
        
        
        unique_arrival_times = sorted(filtered_df['arrival_time'].unique())
        

        start_time = time.time()
        previous_time = 0

       
        events_at_zero = filtered_df[filtered_df['arrival_time'] == 0]
        for _, row in events_at_zero.iterrows():
            executor.handle_event(row)

        for arrival_time in unique_arrival_times:
            if arrival_time == 0:
                continue

            sleep_time = arrival_time - previous_time
            previous_time = arrival_time

            time.sleep(sleep_time)

            if time.time() - start_time > threshold_stop:
                print(f"Breaking after {threshold_stop} seconds (before executing {arrival_time}s events).")
                break

            events_at_time = filtered_df[filtered_df['arrival_time'] == arrival_time]
            for _, row in events_at_time.iterrows():
                executor.handle_event(row)
                print(row)
                

class NormalExecution(ExecutionStrategy):
    def execute(self, simulator):
        filtered_df = simulator.filtered_df
        executor = simulator.executor

        # Get the first row for each unique category
        unique_df = filtered_df.drop_duplicates(subset="func", keep="first")

        print(unique_df)
        exit()
        for _, row in unique_df.iterrows():
            executor.handle_event(row)

        # Execute all events at once
        for _, row in filtered_df.iterrows():
            executor.handle_event(row)
        
    

class TimeBasedExecution(ExecutionStrategy):
    def execute(self, simulator):
        filtered_df = simulator.filtered_df
        executor = simulator.executor
        threshold_stop = simulator.threshold_stop

        unique_arrival_times = sorted(filtered_df['arrival_time'].unique())
        start_time = time.time()
        previous_time = 0
        
        events_at_zero = filtered_df[filtered_df['arrival_time'] == 0]
        for _, row in events_at_zero.iterrows():
            executor.handle_event(row)

        for arrival_time in unique_arrival_times:
            if arrival_time == 0:
                continue

            sleep_time = arrival_time - previous_time
            previous_time = arrival_time

            time.sleep(sleep_time)

            if time.time() - start_time > threshold_stop:
                print(f"Breaking after {threshold_stop} seconds (before executing {arrival_time}s events).")
                break

            events_at_time = filtered_df[filtered_df['arrival_time'] == arrival_time]
            for _, row in events_at_time.iterrows():
                executor.handle_event(row)


class Simulator:
    def __init__(self, filtered_df, executor, strategy, threshold_stop=1050):
        self.filtered_df = filtered_df
        self.executor = executor
        self.threshold_stop = threshold_stop
        self.strategy = strategy  # ExecutionStrategy instance

    def simulate(self):
        self.strategy.execute(self)
