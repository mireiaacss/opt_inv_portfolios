#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd 
import time 
import abc

def load_data( ):
    # Implement a function to load all data except the first row and first column in your code.
    # We strongly encourage you to store your data in a NumPy array.

    data = np.genfromtxt("SP500_data.csv", delimiter=",", skip_header=1) # Load the data from the CSV file
    data = data[:, 1:] # Remove the first column
  
    # return the data
    return data



def aggregate_data( data ):
    # Aggregate the stock value of each company computing the LogRatio in blocks of 7 days (approx. a week).
    # Note that the stock markets close during weekends and bank holidays.

    
    num_days, num_companies = data.shape # Get the dimensions of the data

    num_weeks = num_days // 5 # Calculate the number of weeks in the data
   
    data_week = np.zeros((num_weeks, num_companies)) # Initialise the data_week array

    for week in range(num_weeks): # Loop through each week

        start_idx = week * 5 # Calculate the start index of the week
        end_idx = start_idx + 5 # Calculate the end index of the week

        if end_idx <= num_days: # Check if the end index is within the data
            log_ratio = np.log(data[end_idx-1, :] / data[start_idx, :]) # Calculate the log ratio of the week
            data_week[week, :] = log_ratio # Store the log ratio in the data_week array



    
    # Return the aggregated data called weekly_data
    return data_week



def calculate_mean_std( data ):
    # Compute the aggregated data mean and the standard deviation for each company
    # Store them in a matrix called mean_std containing column-wise.
    # Column 0 should contain all companies' means,
    # and column 1 its corresponding standard deviation
    
    means = np.mean(data, axis=0) # Calculate the mean of the data
    stdv = np.std(data, axis=0) # Calculate the standard deviation of the data
    data_mean_std = np.column_stack((means, stdv)) # Stack the means and standard deviations column-wise



    # Return this matrix mean_std 
    return data_mean_std 



def initial_solution( size, amount = 100 ):
    # Compute the initial solution
    # Giving our capital stored in variable amount (for instance, 100 Meuros)
    # generate the initial asset allocation assigning all money to a random company

    # return the initial solution
    isolution = np.zeros(size) # Initialise the initial solution
    random_company = np.random.randint(0, size) # Generate a random company
    isolution[random_company] = amount # Assign all money to the random company
    
    return isolution # Return the initial solution

def objective_function_VaR( solution, mean_std ):
    # We implement the Monte Carlo method to compute the VaR
    # Giving the mean and standard deviation of each company computed over the aggregated data
    # and the current solution, compute:
    # 1. For each company, draw a random value following a normal distribution with its mean and standard deviation.
    # 2. Use these random values and the current assets allocated to each company to compute the total gain
    # 3. Repeat 100 times points 1 and 2. 
    # Compute the 95% one-tailed Value at Risk over the 100 total gains

    

    num_rep = 100  # Number of repetitions

    random_returns = np.random.normal(loc=mean_std[:, 0], scale=mean_std[:, 1], size=(num_rep, len(solution))) # Draw random values from the normal distribution
    total_gains = random_returns.dot(solution) # Compute the total gain
    VaR_95 = np.percentile(total_gains, 5) # Compute the 95% one-tailed Value at Risk


 
    # Return the VaR
    return sum(solution) + VaR_95




#--------------------------------------------
# Second Session
#--------------------------------------------



def objective_function_sharp( solution, mean_std ):
    
    # Giving the mean and standard deviation of each company computed over the aggregated data
    # and the current solution compute:
    # 1. For each company, draw a random value following a normal distribution with its mean and standard deviation.
    # 2. Use these random values and the current assets allocated to each company to compute the total gain
    # 3. Repeat 100 times points 1 and 2. 
    # Compute the (adapted) Sharp Ratio over the 100 total gains
    # 1. Compute the mean of the gains
    # 2. Compute the standard deviation of the gains
    # 3. Apply the adapted formula for the Sharp Ratio
 
    # Return the sharp ratio

    num_rep = 100 # Number of repetitions

    random_returns = np.random.normal(loc=mean_std[:, 0], scale=mean_std[:, 1], size=(num_rep, len(solution))) # Draw random values from the normal distribution
    
    gain_returns = np.dot(random_returns, solution) # Compute the total gain
    
    mean_of_gains = np.mean(gain_returns) # Compute the mean of the gains
    stdev = np.std(gain_returns) # Compute the standard deviation of the gains
    
    sharp = mean_of_gains / stdev # Compute the Sharp Ratio



    
    return sharp # Return the Sharp Ratio

def objective_function_mdd( solution, mean_std ):
    
    # Giving the mean and standard deviation of each company computed over the aggregated data
    # and the current solution, compute:
    # 1. For each company, draw a random value following a normal distribution with its mean and standard deviation.
    # 2. Use these random values and the current assets allocated to each company to compute the total gain
    # 3. Repeat 100 times points 1 and 2. 
    # Compute the (adapted) maximum drawdow (mdd) over the 100 total gains
    # 1. Find the minimum gain
    # 2. Find the maximum gain
    # 3. Apply the adapted formula for mdd
    
    # Return the mdd

    num_rep = 100 # Number of repetitions
    
    random_returns = np.random.normal(loc=mean_std[:, 0], scale=mean_std[:, 1], size=(num_rep, len(solution))) # Draw random values from the normal distribution
    
    gain_returns = np.dot(random_returns, solution) # Compute the total gain
    
    mdd = (np.min(gain_returns) - np.max(gain_returns)) / np.max(gain_returns) # Compute the maximum drawdown







    
    return mdd # Return the maximum drawdown


def get_neighbour(solution):

    neighbour = solution.copy() # Copy the solution

    companies_with_capital = np.where(neighbour > 0)[0] # Get the indices of the companies with capital

    if len(companies_with_capital) == 0: # If there are no companies with capital
        return neighbour # Return the solution
    
    source_company = np.random.choice(companies_with_capital) # Choose a random company with capital
    all_companies = np.arange(len(neighbour)) # Get all the companies
    destination_company = np.random.choice(all_companies) # Choose a random company

    amount_to_move = np.random.uniform(0.05, 0.1) * neighbour[source_company] # Calculate the amount to move

    neighbour[source_company] -= amount_to_move # Subtract the amount to move from the source company
    neighbour[destination_company] += amount_to_move # Add the amount to move to the destination company

    return neighbour # Return the neighbour




def simulated_annealing_algorithm( data, mean_std, solution, temperature, alpha, num_iter, objective_function ):
    # Optimisation based on Simulated Annealing
    # The objective function is passed as an input so this algorithm can call whatever objective function
    
    # Only one function is defined in this template, but you can create more functions if needed
    # to structure better your code
   
    # Evaluate the initial solution
    # Store the solution as the first best solution together with its evaluation
    # Set the initial temperature
    # Run the optimization loop tracking the current solution and the best asset allocation

    # Use conventional algorithm and temperature updates
    # if the initial solution is not improved at the end of the iterations,
    # try different methods (double loops in SA, different temperature updates, etc.)
    
    # Return the best solution and its evaluation

    current_solution = solution # Set the current solution
    current_eval = objective_function(current_solution, mean_std) # Evaluate the current solution
    best = current_solution.copy() # Copy the current solution
    best_eval = current_eval # Set the best evaluation

    for _ in range(num_iter): # Loop through the iterations
        neighbour_solution = get_neighbour(current_solution) # Get the neighbour solution
        neighbour_eval = objective_function(neighbour_solution, mean_std) # Evaluate the neighbour solution

        delta = neighbour_eval - current_eval # Calculate the difference between the neighbour and current evaluation

        # Add a safety check to prevent overflow in the exponential calculation
        if delta > 0 or (temperature > 1e-10 and np.random.rand() < np.exp(min(-delta / temperature, 700))):
            current_solution = neighbour_solution # Set the current solution to the neighbour solution
            current_eval = neighbour_eval # Set the current evaluation to the neighbour evaluation

        if current_eval > best_eval: # If the current evaluation is better than the best evaluation
            best = current_solution.copy() # Copy the current solution
            best_eval = current_eval # Set the best evaluation to the current evaluation
        
        temperature *= alpha # Cool the temperature

    return best, best_eval # Return the best solution and its evaluation


def tabu_list_management(tabu_list, solution, current_iteration, current_tabu_tenure):
    solution_key = tuple(solution.round(decimals=4)) # Round the solution to 4 decimal places and convert to a tuple
    tabu_list[solution_key] = current_iteration + current_tabu_tenure # Update the tabu list
    tabu_list = {sol: exp for sol, exp in tabu_list.items() if exp > current_iteration} # Remove entries that have expired
    return tabu_list # Return the updated tabu list

def evaluate_neighbours(neighbours, objective_function, mean_std):
    best_neig_eval = float('-inf') # Initialise the best neighbour evaluation
    best = None # Initialise the best neighbour

    for nei in neighbours: # Loop through the neighbours
        current_eval = objective_function(nei, mean_std) # Evaluate the neighbour using the objective function
        if current_eval > best_neig_eval: # If the current evaluation is better than the best neighbour evaluation
            best_neig_eval = current_eval # Set the best neighbour evaluation to the current evaluation
            best = nei # Set the best neighbour to the current neighbour
    
    return best # Return the best neighbour

def tabu_search_algorithm( data, mean_std, solution, tabu_tenure, num_iter, objective_function, num_neighbours = 100, aspiration_threshold = 0.05):
   # Optimisation based on Tabu Search
   # The objective function is passed as an input so this algorithm can call whatever objective function
   
   # Only one function is defined in this template, but you can create more functions if needed
   # to structure better your code
  
   # Evaluate the initial solution
   # Store the solution as the first best solution together with its evaluation
   # Run the optimization loop tracking the current solution and the best asset allocation

   # Use conventional algorithm and aspiration criteria
   # if the initial solution is not improved at the end of the iterations,
   # try different methods (different tabu tenure, different aspiration criteria, etc.)
   
   # Return the best solution and its evaluation
    tabu_list = {} # Initialise the tabu list
    best = solution.copy() # Copy the initial solution
    best_eval = objective_function(solution, mean_std) # Evaluate the initial solution using the objective function

    current_tabu_tenure = tabu_tenure # Set the current tabu tenure
    iterations_without_improvement = 0 # Initialise the number of iterations without improvement

    for i in range(num_iter): # Loop through the iterations
        neighbours = [] # Initialise the neighbours list
        neighbours_eval = [] # Initialise the neighbours evaluation list

        for j in range(num_neighbours): # Loop through the neighbours
            neighbour = get_neighbour(best) # Get the neighbour
            candidate_key = tuple(neighbour.round(decimals=4)) # Round the neighbour to 4 decimal places and convert to a tuple
            candidate_eval = objective_function(neighbour, mean_std) # Evaluate the neighbour using the objective function

            if candidate_key in tabu_list: # If the neighbour is in the tabu list
                if candidate_eval > best_eval * (1 + aspiration_threshold): # If the neighbour is better than the aspiration threshold
                    neighbours.append(neighbour) # Add the neighbour to the neighbours list
                    neighbours_eval.append(candidate_eval) # Add the neighbour evaluation to the neighbours evaluation list
            else: # If the neighbour is not in the tabu list
                neighbours.append(neighbour) # Add the neighbour to the neighbours list
                neighbours_eval.append(candidate_eval) # Add the neighbour evaluation to the neighbours evaluation list
                
        if not neighbours: # If there are no neighbours
            solution = initial_solution(len(best)) # Perform a random restart to allow for diversification
            current_eval = objective_function(solution, mean_std) # Evaluate the new solution using the objective function
        else: # If there are neighbours
            best_idx = np.argmax(neighbours_eval) # Get the index of the best neighbour
            solution = neighbours[best_idx] # Set the solution to the best neighbour
            current_eval = neighbours_eval[best_idx] # Set the current evaluation to the evaluation of the best neighbour
        
        if current_eval > best_eval: # If the current evaluation is better than the best evaluation
            best = solution.copy() # Copy the solution
            best_eval = current_eval # Set the best evaluation to the current evaluation
            iterations_without_improvement = 0 # Reset the number of iterations without improvement
        else: # If the current evaluation is not better than the best evaluation
            iterations_without_improvement += 1 # Increment the number of iterations without improvement
        
        tabu_list = tabu_list_management(tabu_list, solution, i, current_tabu_tenure) # Update the tabu list

        if iterations_without_improvement >= 10: # If the number of iterations without improvement is greater than 10
            current_tabu_tenure = min(current_tabu_tenure + 1, 20) # Increment the tabu tenure
        else: # If the number of iterations without improvement is less than 10
            current_tabu_tenure = tabu_tenure # Reset the tabu tenure
    

    # Return the best solution and its evaluation
    return best, best_eval

#--------------------------------------------
# Main for Second Session
#--------------------------------------------

# Constants
amount = 100
repetitions = 100   # Number of generated random values for the calculation
                    # of the objective functions

num_iter = 100       # Main loop of the simulated annealing

temperature = amount * 1.1   # Initial temperature for the SA
alpha = 0.95                 # Cooling parameter for the SA

tabu_tenure = 10     # Initial value of all elements in the tabu list

np.random.seed(0) # For reproducibility, i.e.,
                  # different executions of the same code
                  # will generate the same random numbers
                  # Keep it to check if the code works
                  # Comment it to check it the code is able
                  # to optimise any combination of randon numbers

data = load_data()
print( "Data shape ", data.shape)

data_week = aggregate_data(data)
print( "Weekly data shape ", data_week.shape)

data_mean_std = calculate_mean_std( data_week )

isolution = initial_solution( data_week.shape[1], amount )
print( "Initial selected companies: \n", isolution )

VaR_95 = objective_function_VaR( isolution, data_mean_std )
sharp = objective_function_sharp( isolution, data_mean_std )
mdd = objective_function_mdd( isolution, data_mean_std )

print(f"\nValue at Risk (VaR) at 95% confidence level: ${VaR_95:,.2f}")
print(f"\nSharp: {sharp:,.2f}")
print(f"\nMaximum Drawdown: {mdd:,.2f}")

# To be completed with the calls to the SA and to the TS uwing the different objective functions
# The recommendation is to use the objective function as an input parameter of the algorithm


#Simulated Annealing

# Using the Value at Risk (VaR) objective function
best_sa_var, best_sa_eval_var = simulated_annealing_algorithm(data_week, data_mean_std, isolution, temperature, alpha, num_iter, objective_function_VaR)

# Using the adapted Sharpe Ratio objective function
best_sa_sharp, best_sa_eval_sharp = simulated_annealing_algorithm(data_week, data_mean_std, isolution, temperature, alpha, num_iter, objective_function_sharp)


# Using the adapted Maximum Drawdown (MDD) objective function
best_sa_mdd, best_sa_eval_mdd = simulated_annealing_algorithm(data_week, data_mean_std, isolution, temperature, alpha, num_iter, objective_function_mdd)


#Tabu Search

# Using the Value at Risk (VaR) objective function
best_ts_var, best_ts_eval_var = tabu_search_algorithm(data_week, data_mean_std, isolution, tabu_tenure, num_iter, objective_function_VaR)


# Using the adapted Sharpe Ratio objective function
best_ts_sharp, best_ts_eval_sharp = tabu_search_algorithm(data_week, data_mean_std, isolution, tabu_tenure, num_iter, objective_function_sharp)


# Using the adapted Maximum Drawdown (MDD) objective function
best_ts_mdd, best_ts_eval_mdd = tabu_search_algorithm(data_week, data_mean_std, isolution, tabu_tenure, num_iter, objective_function_mdd)


#Print the results
print("Simulated Annealing Results:")
print("Best VaR Evaluation: ", best_sa_eval_var)
print("Best Sharpe Evaluation: ", best_sa_eval_sharp)
print("Best MDD Evaluation: ", best_sa_eval_mdd)
print("\nTabu Search Results:")
print("Best VaR Evaluation: ", best_ts_eval_var)
print("Best Sharpe Evaluation: ", best_ts_eval_sharp)
print("Best MDD Evaluation: ", best_ts_eval_mdd)



#--------------------------------------------
# Third Session
#--------------------------------------------

class OptimisationResults(abc.ABC): # Abstract base class for optimisation results
    def __init__(self, columns=[], objective_functions=[], path='', amount=100, repetitions_per_run=10, seed=0): # Initialise the class
        """
        This is the base class for the simulated annealing and tabu search algorithms.
        It is used to store the results of the experiments and to save them to an Excel file.
        args:
            columns: list of column names for the different configurations in the output Excel
            objective_functions: list of objective function references to test
            path: path to the Excel file for saving results
            amount: initial capital amount
            repetitions_per_run: number of repetitions per run
            seed: seed for the random number generator
        
        returns:
            None
        """
        
        self.amount = amount # Initial capital amount
        self.repetitions_per_run = max(1, repetitions_per_run) # Number of repetitions per run
        self.columns = columns # List of column names for the different configurations in the output Excel
        self.objective_functions = objective_functions # List of objective function references to test
        self.path = path # Path to the Excel file for saving results

        if seed is not None: # If a seed is provided
            np.random.seed(seed) # Set the seed

        self.data = load_data() # Load the data
        self.data_week = aggregate_data(self.data) # Aggregate the data
        self.data_mean_std = calculate_mean_std(self.data_week) # Calculate the mean and standard deviation of the data
        self.isolution = initial_solution(self.data_week.shape[1], self.amount) # Initialise the solution

        self.initial_metrics = {} # Initialise the initial metrics
        for obj_func in self.objective_functions: # Loop through the objective functions
            self.initial_metrics[obj_func.__name__] = obj_func(self.isolution.copy(), self.data_mean_std) # Evaluate the initial solution

        self.metric_row_map = { # Map the objective functions to the rows in the Excel
            'objective_function_VaR': 'VaR at 95%',
            'objective_function_sharp': 'Sharp',
            'objective_function_mdd': 'MDD'
        }
        self.df = None # Initialise the DataFrame


    def load_or_create_excel(self):
        """
        Loads existing results from an Excel file or creates a new DataFrame if the file doesn't exist.
        Ensures all expected rows and columns are present in the DataFrame.
        
        returns:
            pandas.DataFrame: DataFrame with optimisation results
        """
        try:
            self.df = pd.read_excel(self.path, index_col=0)
            print(f"Loaded existing results from {self.path}")
        except FileNotFoundError:
            print(f"Creating new results DataFrame for {self.path}")
            rows = [self.metric_row_map[obj_func.__name__] for obj_func in self.objective_functions] + ['Execution Time'] # Create the rows
            initial_cols = ['Initial Values'] + self.columns # Create the columns
            self.df = pd.DataFrame(index=rows, columns=initial_cols) # Create the DataFrame
            self.df = self.df.fillna(pd.NA) # Fill the missing values with NA

        expected_cols = ['Initial Values'] + self.columns # Create the expected columns
        for col in expected_cols: # Loop through the expected columns
            if col not in self.df.columns: # If the column does not exist
                self.df[col] = pd.NA # Fill the missing values with NA

        expected_rows = [self.metric_row_map[obj_func.__name__] for obj_func in self.objective_functions] + ['Execution Time'] # Create the expected rows
        for row in expected_rows: # Loop through the expected rows
            if row not in self.df.index: # If the row does not exist
                self.df.loc[row] = pd.NA # Fill the missing values with NA

        for obj_func, initial_value in self.initial_metrics.items(): # Loop through the initial metrics
            row_name = self.metric_row_map.get(obj_func, None) # Get the row name
            if row_name: # If the row name exists
                self.df.loc[row_name, 'Initial Values'] = initial_value # Fill the initial values
        self.df.loc['Execution Time', 'Initial Values'] = '-' # Fill the execution time with '-'
        
        print("Initial DataFrame state:")
        print(self.df)
        return self.df # Return the DataFrame


    def save_results(self, avg_evaluations, avg_times, config_index):
        """
        Saves the average evaluation results and execution times to the DataFrame.
        
        args:
            avg_evaluations: dictionary of average evaluation results for each objective function
            avg_times: dictionary or float with average execution times
            config_index: index of the configuration in the columns list
            
        returns:
            pandas.DataFrame: Updated DataFrame with new results
        """
        if config_index >= len(self.columns): # If the configuration index is out of bounds
            print(f"Warning: Configuration index {config_index} out of bounds for columns {self.columns}. Skipping save.") # Print a warning
            return self.df # Return the DataFrame

        col_name = self.columns[config_index] # Get the column name

        for obj_func, avg_eval in avg_evaluations.items(): # Loop through the average evaluations
            row_name = self.metric_row_map.get(obj_func.__name__, None) # Get the row name
            if row_name: # If the row name exists
                if isinstance(avg_eval, float): # If the average evaluation is a float
                    self.df.loc[row_name, col_name] = f"{avg_eval:.2f}" # Fill the average evaluation
                else: # If the average evaluation is not a float
                    self.df.loc[row_name, col_name] = avg_eval

        
        if isinstance(avg_times, dict): # If the average times is a dictionary
            avg_exec_time = sum(avg_times.values()) / len(avg_times) if avg_times else 0 # Calculate the average execution time
        else: # If the average times is not a dictionary
            avg_exec_time = avg_times # Set the average execution time to the average times

        self.df.loc['Execution Time', col_name] = f"{avg_exec_time:.3f}s" # Fill the execution time
        return self.df # Return the DataFrame

    @staticmethod
    def _exec_time(function, *args, **kwargs):
        """
        Calculates the execution time of a function.
        
        args:
            function: the function to execute
            *args, **kwargs: arguments to pass to the function
            
        returns:
            tuple: (function_result, execution_time)
        """
        start_time = time.perf_counter() # Start the timer
        result = function(*args, **kwargs) # Execute the function
        exec_time = time.perf_counter() - start_time # Calculate the execution time
        return result, exec_time # Return the result and the execution time

    @abc.abstractmethod
    def _run_single_optimisation(self, config_index, objective_function):
        """
        Abstract method to run a single optimisation experiment.
        Must be implemented by subclasses.
        
        args:
            config_index: index of the configuration in the parameters lists
            objective_function: objective function to optimise
            
        returns:
            tuple: ((best_solution, best_evaluation), execution_time)
        """
        pass
        
    def perform_experiments(self):
        """
        Performs all the optimisation experiments according to the configurations.
        For each configuration and objective function, runs multiple repetitions and 
        calculates average evaluation results and execution times.
        Saves results to the Excel file after each configuration is completed.
        
        returns:
            pandas.DataFrame: Final DataFrame with all results
        """
        self.df = self.load_or_create_excel() # Load or create the Excel file
        num_configs = len(self.columns) # Get the number of configurations

        for i in range(num_configs): # Loop through the configurations
            print(f"\n--- Running Configuration: {self.columns[i]} ---") # Print the configuration
            avg_evaluations = {} # Initialise the average evaluations
            average_times = {} # Initialise the average times

            for obj_func in self.objective_functions: # Loop through the objective functions
                print(f"  Optimising for: {obj_func.__name__} ({self.repetitions_per_run} repetitions)") # Print the objective function
                eval_results_list = [] # Initialise the evaluation results list
                exec_times_list = [] # Initialise the execution times list

                for rep in range(self.repetitions_per_run): # Loop through the repetitions
                    (best_sol, best_eval), exec_time = self._run_single_optimisation(i, obj_func) # Run the single optimisation
                    eval_results_list.append(best_eval) # Append the evaluation results
                    exec_times_list.append(exec_time) # Append the execution times
                    print(f"    Rep {rep+1}/{self.repetitions_per_run}: Eval={best_eval:.4f}, Time={exec_time:.4f}s") # Print the evaluation and execution time

                if eval_results_list: # If the evaluation results list is not empty
                    avg_eval = np.mean(eval_results_list) # Calculate the average evaluation
                    average_time = np.mean(exec_times_list) # Calculate the average time
                    avg_evaluations[obj_func] = avg_eval # Fill the average evaluations
                    average_times[obj_func] = average_time # Fill the average times
                    print(f"  -> Avg Eval: {avg_eval:.4f}, Avg Time: {average_time:.4f}s") # Print the average evaluation and time
                else: # If the evaluation results list is empty
                    avg_evaluations[obj_func] = pd.NA # Fill the average evaluations with NA

            self.df = self.save_results(avg_evaluations, average_time, i) # Save the results

            try:
                 self.df.to_excel(self.path) # Save the results to the Excel file
                 print(f"Saved intermediate results to {self.path}") 
            except Exception as e:
                 print(f"Error saving Excel file: {e}") # Print an error message


        print("\n--- Experiment Complete ---")
        print("Final Results DataFrame:")
        print(self.df)
        return self.df # Return the DataFrame

# --- Subclass for Simulated Annealing ---
class Results_Simulated_Annealing(OptimisationResults):
    def __init__(self, num_iters=[100], temp=[110], alpha=[0.95], **kwargs):
        """
        Subclass for running Simulated Annealing optimisation experiments.
        
        args:
            num_iters: list of number of iterations for each configuration
            temp: list of initial temperatures for each configuration
            alpha: list of cooling rates for each configuration
            **kwargs: arguments to pass to the parent class
            
        returns:
            None
        """
        super().__init__(**kwargs) # Initialise the parent class
        self.num_iters = num_iters # Fill the number of iterations
        self.temp = temp # Fill the temperature
        self.alpha = alpha # Fill the alpha

        n_cols = len(self.columns) # Get the number of columns
        if len(self.num_iters) != n_cols or len(self.temp) != n_cols or len(self.alpha) != n_cols: # If the lengths of the parameter lists do not match the number of columns
             print("Warning: Lengths of SA parameter lists (num_iters, temp, alpha) do not match the number of columns.") # Print a warning


    def _get_params(self, config_index): # Get the parameters
        
        iters = self.num_iters[config_index] if config_index < len(self.num_iters) else self.num_iters[-1] # Get the number of iterations
        temp_val = self.temp[config_index] if config_index < len(self.temp) else self.temp[-1] # Get the temperature
        alpha_val = self.alpha[config_index] if config_index < len(self.alpha) else self.alpha[-1] # Get the alpha
        return iters, temp_val, alpha_val # Return the number of iterations, temperature and alpha


    def _run_single_optimisation(self, config_index, objective_function): # Run the single optimisation
        
        iters, temp_val, alpha_val = self._get_params(config_index) # Get the parameters

        solution_copy = self.isolution.copy() # Copy the solution

        result, exec_time = self._exec_time(
            simulated_annealing_algorithm, # Call the simulated annealing algorithm
            self.data_week,
            self.data_mean_std,
            solution_copy, 
            temp_val,
            alpha_val,
            iters,
            objective_function
        )
        return result, exec_time # Return the result and the execution time

# --- Subclass for Tabu Search ---
class Results_Tabu_Search(OptimisationResults):
    def __init__(self, num_iters=[100], num_neighbours=[50], tabu_tenure=[10], aspiration_threshold=[0.05], **kwargs):
        """
        Subclass for running Tabu Search optimisation experiments.
        
        args:
            num_iters: list of number of iterations for each configuration
            num_neighbours: list of number of neighbours to evaluate for each configuration
            tabu_tenure: list of tabu tenures for each configuration
            aspiration_threshold: list of aspiration thresholds for each configuration
            **kwargs: arguments to pass to the parent class
            
        returns:
            None
        """
        super().__init__(**kwargs) # Initialise the parent class
        self.num_iters = num_iters # Fill the number of iterations
        self.num_neighbours = num_neighbours # Fill the number of neighbours
        self.tabu_tenure = tabu_tenure # Fill the tabu tenure
        self.aspiration_threshold = aspiration_threshold # Fill the aspiration threshold
        
        n_cols = len(self.columns) # Get the number of columns
        if len(self.num_iters) != n_cols or len(self.num_neighbours) != n_cols or len(self.tabu_tenure) != n_cols: # If the lengths of the parameter lists do not match the number of columns
            print("Warning: Lengths of TS parameter lists (num_iters, num_neighbours, tabu_tenure) do not match the number of columns.") # Print a warning
        

    def _get_params(self, config_index): # Get the parameters
        
        iters = self.num_iters[config_index] if config_index < len(self.num_iters) else self.num_iters[-1] # Get the number of iterations
        neighbours = self.num_neighbours[config_index] if config_index < len(self.num_neighbours) else self.num_neighbours[-1] # Get the number of neighbours   
        tenure = self.tabu_tenure[config_index] if config_index < len(self.tabu_tenure) else self.tabu_tenure[-1] # Get the tabu tenure
        threshold = self.aspiration_threshold[config_index] if config_index < len(self.aspiration_threshold) else self.aspiration_threshold[-1] # Get the aspiration threshold
        return iters, neighbours, tenure, threshold # Return the number of iterations, neighbours, tenure and aspiration threshold


    def _run_single_optimisation(self, config_index, objective_function): # Run the single optimisation
        
        iters, neighbours_val, tenure_val, threshold_val = self._get_params(config_index) # Get the parameters

       
        solution_copy = self.isolution.copy() # Copy the solution

        result, exec_time = self._exec_time(
            tabu_search_algorithm, # Call the tabu search algorithm
            self.data_week,
            self.data_mean_std,
            solution_copy, 
            tenure_val,
            iters,
            objective_function,
            num_neighbours=neighbours_val,
            aspiration_threshold=threshold_val
        )
        return result, exec_time # Return the result and the execution time


# --- Main Execution Block Session 3 ---
if __name__ == "__main__":

    #Inside of the folder there is an empty template that is tailored to the needs of the simulated annealing and tabu search algorithms classes.
    #I will leave it empty if there is a need to run the simulations again.

    # Define common parameters
    output_excel_file = 'Results_improvements_refactored.xlsx'
    objective_functions_to_run = [objective_function_VaR, objective_function_sharp, objective_function_mdd]
    repetitions_for_average = 10 

    # --- Simulated Annealing Experiments ---
    print("="*20 + " Running Simulated Annealing " + "="*20)
    sa_num_iters =      [100,   1000,   100,    1000,   1000] # Base, Impr-1, Impr-2, Impr-3, Impr-4
    sa_temp =           [110,   110,    140,    110,    150]
    sa_alpha =          [0.95,  0.95,   0.95,   0.99,   0.99]
    sa_columns = ['Simulated Annealing Base Case', 'Simulated Annealing Improv-1', 'Simulated Annealing Improv-2', 'Simulated Annealing Improv-3', 'Simulated Annealing Improv-4']

    sa_runner = Results_Simulated_Annealing(
        num_iters=sa_num_iters,
        temp=sa_temp,
        alpha=sa_alpha,
        columns=sa_columns,
        objective_functions=objective_functions_to_run,
        path=output_excel_file,
        repetitions_per_run=repetitions_for_average,
        seed=0 
    )
    final_sa_df = sa_runner.perform_experiments()
    print("\nSimulated Annealing Final Results:")

    display_cols_sa = ['Initial Values'] + sa_columns
    print(final_sa_df[display_cols_sa])


    # --- Tabu Search Experiments ---
    print("\n" + "="*20 + " Running Tabu Search " + "="*20)
    ts_num_iters =      [100,  1000,   100,   100,   100] # Base, Impr-1, Impr-2, Impr-3, Impr-4
    ts_num_neighbours = [10,   10,    50,   10,   10] 
    ts_tabu_tenure =    [10,   10,    10,    30,    10]
    ts_aspiration_threshold = [0.05, 0.05, 0.05, 0.05, 0.03]
    ts_columns = ['Tabu Search Base Case', 'Tabu Search Improv-1', 'Tabu Search Improv-2', 'Tabu Search Improv-3', 'Tabu Search Improv-4']

    ts_runner = Results_Tabu_Search(
        num_iters=ts_num_iters,
        num_neighbours=ts_num_neighbours,
        tabu_tenure=ts_tabu_tenure,
        aspiration_threshold=ts_aspiration_threshold,
        columns=ts_columns,
        objective_functions=objective_functions_to_run,
        path=output_excel_file,
        repetitions_per_run=repetitions_for_average,
        seed=0 
    )
    final_ts_df = ts_runner.perform_experiments() 
    print("\nTabu Search Final Results (including previous SA runs):")
    print(final_ts_df)










