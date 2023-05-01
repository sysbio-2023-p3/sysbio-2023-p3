import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import collections
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# has to read in data. Format:
    # From python: 
        # two column csv with 'times' and 'reps', each an individual 
        # commited cell
        # csv with columns for time, state0, state1, state2, and state3 (in that order)
    # From matlab: 
        # single csv with columns for state0, state1, state2, state3 and state4 (commited)
        # and time (in that order)
        
# always include file extensions in arguments
def visualise_python_model(times_reps_csv, states_csv, save=True, do_reps_hexbin=False,
                     hexbin_name="default_hexbin.png", do_counts_time=False,
                     counts_time_name_16="default_counts_16_scatter.png",
                     counts_time_name_total="default_counts_total_scatter.png",
                     do_log_counts_time=False,
                     log_counts_time_name_16="default_counts_log_16_scatter.png",
                     log_counts_time_name_total="default_counts_log_total_scatter.png",
                     do_states=False, states_name_averages="default_states_averages.png", \
                     states_name_days="default_states_days.png", log_states=True):
    times_reps_df = pd.read_csv(times_reps_csv)
    states_df = pd.read_csv(states_csv)
    
    if do_reps_hexbin == True:
        hex_fig = plt.figure()
        plt.hexbin(times_reps_df["times"], times_reps_df["reps"])
        plt.xlabel("Time since puberty onset (days)")
        plt.ylabel("Number of cell cycles (since puberty onset) before commitment")
        plt.tight_layout()
        plt.show()
        if save == True:
            hex_fig.savefig(hexbin_name)
        
    if do_counts_time == True:
        # should only be used if cycle lengths for all states were the same
        cycle_counts_dict = collections.Counter(np.array(times_reps_df["times"]))
        days = list(cycle_counts_dict.keys())
        counts = list(cycle_counts_dict.values())
        years = list(np.floor(np.array(days)/365))
        count_year_df = pd.DataFrame(zip(years, counts), columns=["year","count"])
        
        # average number of committed cells produced in each 16 day cycle for each year
        fig, ax = plt.subplots()
        ax = sns.lineplot(count_year_df, x="year", y="count", errorbar=("sd", 2))
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Average number of cells commiting to differentiation in each 16 day cycle")
        plt.tight_layout()
        fig.show()
        if save == True:
            plt.savefig(counts_time_name_16)
        
        # total committed cell count for each year
        years_counts_dict = collections.Counter(np.floor(np.array(times_reps_df["times"])/365))
        fig, ax = plt.subplots()
        ax.scatter(list(years_counts_dict.keys()), list(years_counts_dict.values()))
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Total number of cells commiting to differentiation per year")
        plt.tight_layout()
        plt.show()
        if save == True:
            plt.savefig(counts_time_name_total)
    
    if do_log_counts_time == True:
        # average (mean) 16 day count for each year
        fig, ax = plt.subplots()
        ax = sns.lineplot(count_year_df, x="year", y="count", errorbar=("sd", 2))
        ax.set(yscale="log")
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Average number of cells commiting to differentiation in each 16 day cycle")
        plt.tight_layout()
        fig.show()
        if save == True:
            plt.savefig(log_counts_time_name_16)
        
        # total count for each year
        fig, ax = plt.subplots()
        ax.scatter(list(years_counts_dict.keys()), list(years_counts_dict.values()))
        ax.set(yscale="log")
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Total number of cells commiting to differentiation per year")
        plt.tight_layout()
        plt.show()
        if save == True:
            plt.savefig(log_counts_time_name_total)
    
    if do_states==True:
        # average population for each year
        # add years column to states_df
        states_df["years"] = np.floor(np.array(states_df["time"])/365)
        fig = sns.relplot(states_df.melt(id_vars=["time", "years"], \
                                         value_vars=["state0", "state1", "state2", "state3"]), \
                          x="years", y="value", hue="variable", kind="line", \
                          palette="colorblind", errorbar=("sd", 2))
        if log_states==True:
            fig.set(yscale="log")
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Average number of cells in each state by year")
        plt.show()
        if save == True:
            plt.savefig(states_name_averages)
        
        fig = sns.relplot(states_df.melt(id_vars=["time", "years"], \
                                         value_vars=["state0", "state1", "state2", "state3"]), \
                          x="time", y="value", hue="variable", kind="line", \
                          palette="colorblind", errorbar=("sd", 2))
        if log_states==True:
            fig.set(yscale="log")
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Number of cells in each state")
        plt.tight_layout()
        plt.show()
        if save == True:
            plt.savefig(states_name_days)


# only use do_log_counts_time=True if do_counts_time=True
def visualise_gillespie_model(read_csv, save=True, do_counts_time=False,
                              counts_time_name_total="default_counts_total_scatter.png",
                              counts_time_16_name="default_counts_time_16.png",
                              do_log_counts_time=False,
                              log_counts_time_name_total="default_counts_log_total_scatter.png",
                              log_counts_time_name_16="default_counts_log_16_scatter.png",
                              do_states=False, states_name="default_states.png",
                              do_log_states=False, log_states_name="default_log_states.png"):

    time_states_df = pd.read_csv(read_csv, header=None, names=["state0", "state1", "state2", \
                                                               "state3", "commited", "time"])
    
    if do_counts_time==True:
        year_cutoffs = np.array(range(1,81))*365
        cumulative_counts = [0]*80
        per_year_counts = [0]*80
        for i in range(len(year_cutoffs)):
            cumulative_counts[i] = max(time_states_df.loc[time_states_df["time"]<year_cutoffs[i], "commited"])
            if i > 0:
                per_year_counts[i] = cumulative_counts[i] - cumulative_counts[i-1]
            else:
                per_year_counts[i] = cumulative_counts[i]
        fig, ax = plt.subplots()
        ax.scatter(list(range(1,81)), list(per_year_counts))
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Total number of cells commiting to differentiation per year")
        plt.tight_layout()
        plt.show()
        if save == True:
            plt.savefig(counts_time_name_total)
        
        # 16 day groups to match program A model format
        cycle_cutoffs = range(16, 29200, 16)
        cumulative_cycle_counts = [0]*len(cycle_cutoffs)
        per_cycle_counts = [0]*len(cycle_cutoffs)
        for i in range(len(cycle_cutoffs)):
            print(i)
            cumulative_cycle_counts[i] = max(time_states_df.loc[time_states_df["time"]<cycle_cutoffs[i], "commited"])
            if i > 0:
                per_cycle_counts[i] = cumulative_cycle_counts[i] - cumulative_cycle_counts[i-1]
            else:
                per_cycle_counts[i] = cumulative_cycle_counts[i]
        cycles_df = pd.DataFrame()
        cycles_df["year"] = np.floor(np.array(cycle_cutoffs)/365)
        cycles_df["count"] = per_cycle_counts
        fig, ax = plt.subplots()
        ax = sns.lineplot(cycles_df, x="year", y="count", errorbar=("sd", 2))
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Average number of cells commiting to differentiation in each 16 day cycle")
        plt.tight_layout()
        fig.show()
        if save == True:
            plt.savefig(counts_time_16_name)
        
        
        
    if do_log_counts_time==True:
        fig, ax = plt.subplots()
        ax.scatter(list(range(1,81)), list(per_year_counts))
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Total number of cells commiting to differentiation per year")
        ax.set(yscale="log")
        plt.tight_layout()
        plt.show()
        if save == True:
            plt.savefig(log_counts_time_name_total)
            
        # 16 log
        fig, ax = plt.subplots()
        ax = sns.lineplot(cycles_df, x="year", y="count", errorbar=("sd", 2))
        ax.set(yscale="log")
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Average number of cells commiting to differentiation in each 16 day cycle")
        plt.tight_layout()
        fig.show()
        if save == True:
            plt.savefig(log_counts_time_name_16)
            
            
    if do_states==True:
        time_states_df["years"] = np.floor(np.array(time_states_df["time"])/365)
        # plots mean of all points at the same time (years)
        fig = sns.relplot(time_states_df.melt(id_vars=["time", "years", "commited"], \
                                         value_vars=["state0", "state1", "state2", "state3"]), \
                          x="years", y="value", hue="variable", kind="line", \
                          palette="colorblind", errorbar=("sd", 2))
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Average number of cells in each state each year")
        plt.show()
        if save == True:
            plt.savefig(states_name)
            
    if do_log_states==True:
        time_states_df["years"] = np.floor(np.array(time_states_df["time"])/365)
        # plots mean of all points at the same time (years)
        fig = sns.relplot(time_states_df.melt(id_vars=["time", "years", "commited"], \
                                         value_vars=["state0", "state1", "state2", "state3"]), \
                          x="years", y="value", hue="variable", kind="line", \
                          palette="colorblind", errorbar=("sd", 2))
        plt.xlabel("Time since puberty onset (years)")
        plt.ylabel("Average number of cells in each state each year")
        fig.set(yscale="log")
        plt.show()
        if save == True:
            plt.savefig(log_states_name)

