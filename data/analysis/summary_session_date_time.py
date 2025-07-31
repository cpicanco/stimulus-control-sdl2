import os
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
from fileutils import cache_folder

def get_unique_days_per_participant(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, sep='\t', header=None, names=['participant', 'date'])

    # Group by participant and get unique dates for each
    unique_days = df.groupby('participant')['date'].unique().to_dict()

    # Convert numpy arrays to lists for better readability
    for participant in unique_days:
        unique_days[participant] = list(unique_days[participant])

    return unique_days

def calculate_days_between_successive_days(unique_days, statistics=max):
    max_days_between = {}
    for participant, dates in unique_days.items():
        # Convert date strings to datetime objects
        date_objs = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

        # Sort the dates
        date_objs.sort()

        # Calculate differences between consecutive days
        if len(date_objs) > 1:
            differences = [(date_objs[i+1] - date_objs[i]).days + 1 for i in range(len(date_objs)-1)]
            max_diff = statistics(differences)
        else:
            max_diff = 0  # Only one date, so no difference

        max_days_between[participant] = max_diff

    # # print max days between days per participant
    # for participant in max_days_between:
    #     print(f"Days between successive days for {participant}: {max_days_between[participant]}")

    return max_days_between

def calculate_days_between_first_last(unique_days):
    days_difference = {}
    for participant, dates in unique_days.items():
        # Convert date strings to datetime objects
        date_objs = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

        # Sort the dates
        date_objs.sort()

        # Calculate difference between first and last day
        if len(date_objs) > 1:
            diff = (date_objs[-1] - date_objs[0]).days + 1
        else:
            diff = 0  # Only one date, so no difference

        days_difference[participant] = diff

    # # Print days between first and last day per participant
    # for participant in days_difference:
    #     print(f"Days between first and last day for {participant}: {days_difference[participant]}")

    return days_difference

def calculate_statistics(num_unique_days):
    # Calculate mean and standard deviation
    mean = np.mean(num_unique_days)
    sd = np.std(num_unique_days, ddof=1)  # ddof=1 for sample standard deviation

    # Calculate 95% confidence interval
    n = len(num_unique_days)
    if n > 30:
        # Use z-score for large samples
        z_score = stats.norm.ppf(0.975)
        se = sd / np.sqrt(n)
        ci_lower = mean - z_score * se
        ci_upper = mean + z_score * se
    else:
        # Use t-score for small samples
        t_score = stats.t.ppf(0.975, df=n-1)
        se = sd / np.sqrt(n)
        ci_lower = mean - t_score * se
        ci_upper = mean + t_score * se

    return {
        'min': min(num_unique_days),
        'max': max(num_unique_days),
        'mean': mean,
        'sd': sd,
        'ci_95': (ci_lower, ci_upper)
    }

def print_stats(stats):
    print(f"Minimum: {stats['min']}")
    print(f"Maximum: {stats['max']}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Standard Deviation: {stats['sd']:.2f}")
    print(f"95% Confidence Interval: ({stats['ci_95'][0]:.2f}, {stats['ci_95'][1]:.2f})")
    print()



cache_foldername = cache_folder()
csv_file_path = os.path.join(cache_foldername, 'session_date_time.csv')

unique_days = get_unique_days_per_participant(csv_file_path)

num_unique_days = [len(days) for days in unique_days.values()]
print("Number of unique days per participant:")
print_stats(calculate_statistics(num_unique_days))

max_days = calculate_days_between_successive_days(unique_days, min)
num_unique_days = [days for days in max_days.values()]
print("Max days between unique days per participant:")
print_stats(calculate_statistics(num_unique_days))

first_last_diff = calculate_days_between_first_last(unique_days)
num_unique_days = [days for days in first_last_diff.values()]
print("Days between first and last day per participant:")
print_stats(calculate_statistics(num_unique_days))