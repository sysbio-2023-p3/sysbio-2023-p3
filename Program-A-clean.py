import numpy as np
from numpy.random import random
import scipy
import collections
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns
import textwrap

# default behaviour; all transitions are basic, no state behaviours except for state 3 (commit)
# horizontal cutoff; maximum factor by which total population in states 0-3 can increase
# before the run is terminated
def population_run(t01="b", t10="b", t12="b", t21="b", t23="b", t32="b", \
                   s0="n", s1="n", s2="n", s3="c", \
                   cycle_times="16;16;16;16", \
                   death_probabilities="0.005;0.005;0.005;0.005", \
                   duplication_probabilities="0.2;0.2;0.2;0.2", \
                   # string format used to enable handling of multiple probability options
                   p01="0.8", p10="0.1", p12="0.8", p21="0.1", p23="0.8", p32="0.1", \
                   population_0=100, population_3=1000, t_s=50, start_reps=30, horizontal_cutoff=10, \
                   write_to_csv=False, times_reps_csv="default_times_reps_file.csv", \
                   states_csv="default_states_file.csv"):
    run_details = "Script: " + __file__ + \
        "\n\npopulation state 0: " + str(population_0) \
        + "\n\npopulation state 3: " + str(population_3) \
        + "\n\nTransitions (type, probability):\n0->1 : " + str(t01) + ", " + str(p01) + "\n1->0 : " + str(t10) + ", " \
        + str(p10) + "\n1->2 : " + str(t12) + ", " + str(p12) + "\n2->1 : " + str(t21) + ", " \
        + str(p21) + "\n2->3 : " + str(t23) + ", " + str(p23) + "\n3->2 : " + str(t32) + ", " \
        + str(p32) + "\n\nState behaviours:\ns0: " + s0 + ", s1: " + s1 + ", s2: " + s2 \
        + ", s3: " + s3 + "\n\nduplication_probabilities: " + duplication_probabilities \
        + "\ndeath_probabilities: " + death_probabilities \
        + "\ncycle_times: " + cycle_times \
        + "\nt_s: " + str(t_s) + ", start_reps: " + str(start_reps)
    
    sperm_times = []
    sperm_reps = []
    states_times_df = pd.DataFrame(columns=["time", "state0", "state1", "state2", "state3"])
    
    cycles = tuple(float(x) for x in cycle_times.split(";"))
    dupes = tuple(float(x) for x in duplication_probabilities.split(";"))
    deaths = tuple(float(x) for x in death_probabilities.split(";"))
    
    transitions = {
        "t01": t01,
        "t10": t10,
        "t12": t12,
        "t21": t21,
        "t23": t23,
        "t32": t32
        }
    
    transition_probs = {
        "p01": p01,
        "p10": p10,
        "p12": p12,
        "p21": p21,
        "p23": p23,
        "p32": p32
        }
    
    states = {
        0: s0,
        1: s1,
        2: s2,
        3: s3,
        }
    
    # default behaviour; state transition without cell division (no dna replication)
    def basic(old_state, new_state, p, ptransition, time, reps):
        ptransition = float(ptransition)
        if p < ptransition:
            # add cycle length of original state to time
            # no change in reps because there hasn't been a cell division
            newest_calls = [[new_state, time + cycles[old_state], reps]]
            return False, -2, newest_calls # tells the calling function that the cell has left the original state
        else:
            p = p - ptransition
            return True, p, [] # tells the calling function that the cell has NOT left the original state, and returns new p
        
    # state transition by asymetrical cell division (divides into one cell of the old state
    # and one cell of the new state)
    def asymmetric(old_state, new_state, p, ptransition, time, reps):
        ptransition = float(ptransition)
        if p < ptransition:
            newest_calls = [[old_state, time + cycles[old_state], reps + 1],
                            [new_state, time + cycles[old_state], reps + 1]]
            return False, -2, newest_calls
        else:
            p = p - ptransition
            return True, p, []
        
    # state transition by symmetrical cell division (divides into two cells of the new state)
    def symmetric(old_state, new_state, p, ptransition, time, reps):
        ptransition = float(ptransition)
        if p < ptransition:
            newest_calls = [[new_state, time + cycles[old_state], reps + 1],
                            [new_state, time + cycles[old_state], reps + 1]]
            return False, -2, newest_calls
        else:
            p = p - ptransition
            return True, p, []
        
    def no_transition(old_state, new_state, p, ptransition, time, reps):
        # no action, returns true because can't leave the state
        return True, p, []
        
    transition_types = {
        "b": basic,
        "a": asymmetric,
        "s": symmetric,
        "n": no_transition
        }
    
    # state functions always happen after all transition probabilities are handled
    # so there is no need to manipulate and return p

    # default state behaviour (no change)
    def none(time, reps, state, p):
        # no extra behaviour, just placeholder to allow alternative behaviour
        # (always returns true because never leaves the original state)
        return True, []

    def duplicate(time, reps, state, p):
        if p < dupes[state]:
            # +1 to reps because cell division (dna replication) has occured
            # cycle length of current cell state added to time
            newest_calls = [[state, time + cycles[state], reps + 1],
                            [state, time + cycles[state], reps + 1]]
            # cell has left original state so returns False
            return False, newest_calls
        else:
            # event hasn't occured so returns True
            return True, []

    # commitment: asymmetrical division into one cell of original state and one 
    # commited to differentiation (recorded)
    def commit(time, reps, state, p):
        sperm_reps.append(reps)
        sperm_times.append(time)
        newest_calls = [[state, time + cycles[state], reps + 1]]
        # always returns false because an action has always happened
        return False, newest_calls

    state_behaviours = {
        "n": none,
        "d": duplicate,
        "c": commit
        }

    def cell_step(state, time, reps):
        
        if time > 29200:
            return [] # empty list won't add anything to the next_calls list of lists
        
        present = True
        p = random()
        
        if p < deaths[state]:
            # set newest_calls to empty list to add nothing to next_calls (no cell/s descending from this one)
            newest_calls = []
            return newest_calls # cell death (won't continue with cell_step if dead)
        p = p - deaths[state]

        # back transition
        if state != 0:
            present, p, newest_calls = transition_types[transitions["t" + str(state) + str(state - 1)]]\
                (state, state-1, p, transition_probs["p" + str(state) + str(state - 1)], time, reps)
        # if a transition has happened, kill cell_step and return newest_calls (arguments for next run/s)
        if present == False:
            return newest_calls        
            
        # forward transition
        if state != 3:
            present, p, newest_calls = transition_types[transitions["t" + str(state) + str(state + 1)]]\
                (state, state+1, p, transition_probs["p" + str(state) + str(state + 1)], time, reps)
        # if a transition has happened, kill cell_step and return newest_calls (arguments for next run/s)        
        if present == False:
            return newest_calls 
                
        # state behaviour
        present, newest_calls = state_behaviours[states[state]](time, reps, state, p)
        if present == True:
            newest_calls = [[state, time + cycles[state], reps]]
            return newest_calls
        else:
            return newest_calls
        
    current_calls = [[0,0,0]] * population_0 + [[3,0,0]] * population_3
    count = 0
    while len(current_calls) <= (population_0+population_3)*horizontal_cutoff and len(current_calls) > 0:
        next_calls = []
        individual_states = [sublist[0] for sublist in current_calls]
        state_counts_dict = collections.Counter(individual_states)
        # assumes time from one cell is the same for all
        states_times_df.loc[len(states_times_df)] = current_calls[0][1], state_counts_dict[0], \
            state_counts_dict[1], state_counts_dict[2], state_counts_dict[3]
        for i in range(len(current_calls)):
            # either a list of lists of the arguments for cell_step call/s for
            # the daughters/next state of the cell , or an empty list if the cell died
            newest_calls = cell_step(current_calls[i][0], current_calls[i][1], current_calls[i][2])
            next_calls.extend(newest_calls)
        current_calls = next_calls
        count += 1
    # if terminated due to 'exploding' len(current_calls) is more than population_0 
    # + population_3.
    if len(current_calls) > (population_0+population_3):
        exploded = True
    else:
        exploded = False
    
    # write to a file IF argument given
    if write_to_csv==True:
        results_to_write_df = pd.DataFrame(list(zip(sperm_times, sperm_reps)), \
                                           columns=["times", "reps"])
        results_to_write_df.to_csv(times_reps_csv, index=False)
        states_times_df.to_csv(states_csv, index=False)
    
    return sperm_times, sperm_reps, run_details, len(current_calls), states_times_df, exploded
    
def score_prerun(sperm_times, sperm_reps, net_stemcell_change, run_details, exploded):
    sperm_array_times = np.array(sperm_times)
    sperm_array_reps = np.array(sperm_reps)
    if len(sperm_times) < 2:
        return "NA", "NA", "NA"
    last_sperm = max(sperm_times)
    last_year = last_sperm / 365
    
    reps_regression = scipy.stats.linregress(sperm_array_times, sperm_array_reps)
    reps_r = reps_regression[2]
    
    sperm_array_years = np.floor(sperm_array_times / 365)
    year_counts_dict = collections.Counter(sperm_array_years)
    if len(year_counts_dict.keys()) > 1:
        array_years = np.array(list(year_counts_dict.keys()))
        array_counts = np.array(list(year_counts_dict.values()))
        counts_regression = scipy.stats.linregress(array_years, array_counts)
        
        # index 0 for slope of linear relationship
        counts_slope = counts_regression[0]
    else:
        counts_slope = "NA"
    
    return last_year, reps_r, counts_slope
    
    
def random_sampling(t01="b", t10="b", t12="b", t21="b", t23="b", t32="b", \
                   s0="n", s1="n", s2="n", s3="c", \
                   cycle_times="16;16;16;16", \
                   death_probabilities="0.1;0.1;0.1;0.1", \
                   duplication_probabilities="0.2;0.2;0.2;0.2", \
                   population_0=100, population_3=1000, t_s=50, start_reps=30, max_p=0.1):
    # assign a random number between 0 and the maximum probability for each transition probability
    (p01, p10, p12, p21, p23, p32) = (np.random.uniform(0, max_p), np.random.uniform(0, max_p), \
                                      np.random.uniform(0, max_p), np.random.uniform(0, max_p), \
                                      np.random.uniform(0, max_p), np.random.uniform(0, max_p))
    # run with the generated set of parameters
    sperm_times, sperm_reps, run_details, net_stemcell_change, states_times_df, exploded = \
        population_run(t01=t01, t10=t10, t12=t12, t21=t21, t23=t23, t32=t32, \
                   s0=s0, s1=s1, s2=s2, s3=s3, \
                   cycle_times=cycle_times, \
                   death_probabilities=death_probabilities, \
                   duplication_probabilities=duplication_probabilities, \
                   p01=p01, p10=p10, p12=p12, p21=p21, p23=p23, p32=p32, \
                   population_0=population_0, population_3=population_3, \
                   t_s=t_s, start_reps=start_reps)
    return sperm_times, sperm_reps, run_details, net_stemcell_change, exploded, \
        p01, p10, p12, p21, p23, p32

# NEWER VERSION FURTHER DOWN

# include .csv extension in argument for file name
# adds argument 'randomise_death', default false, to also randomise death probabilities if true
# one_death generates one random death probability for each run to be shared by all states
# (redundant if randomise_death is false)
# sm_file : where to same the matrix of parameters needed for the classifier
# ca_file : where to save the list of classes needed for the classifier
# survival_cutoff : minimum number of years survival to be classified as successful
def score_random_set(csv_name="run_info_default.csv", runs=10, sm_file="sm_default.csv", ca_file="ca_default.csv", \
                    t01="b", t10="b", t12="b", t21="b", t23="b", t32="b", \
                    s0="n", s1="n", s2="n", s3="c", cycle_times="16;16;16;16", \
                    death_probabilities="0.1;0.1;0.1;0.1", \
                    duplication_probabilities="0.2;0.2;0.2;0.2", \
                    population_0=100, population_3=1000, t_s=50, start_reps=30, randomise_death=False, \
                    one_death=True, death_min_p=0, death_max_p=1, survival_cutoff=10):
    results_df = pd.DataFrame(columns=['p01', 'p10', 'p12', 'p21', 'p23', 'p32', \
                                       'last_year', 'reps_r', 'counts_slope', 'net_stemcell_change', \
                                       'run_details', 's0', 's1', 's2', 's3', 'cycle_times', \
                                       'death_probabilities', 'duplication_probabilities', \
                                       'population_0', 'population_3', 't_s', 'start_reps', \
                                       't01', 't10', 't12', 't21', 't23', 't32', \
                                       'death0', 'death1', 'death2', 'death3'])
    for i in range(runs):
        print(i)
        if randomise_death == True:
            if one_death == True:
                death_p = str(np.random.uniform(death_min_p, death_max_p))
                death_probabilities = death_p + ";" + death_p + ";" + death_p + \
                    ";" + death_p
            else:
                death_probabilities = str(np.random.uniform(death_min_p, death_max_p)) + ";" \
                    + str(np.random.uniform(death_min_p, death_max_p)) + ";" \
                    + str(np.random.uniform(death_min_p, death_max_p)) + ";" \
                    + str(np.random.uniform(death_min_p, death_max_p))
        #print(death_probabilities)
        (death0, death1, death2, death3) = tuple(float(x) for x in death_probabilities.split(";"))
        # run with random parameters 10 seperate times
        sperm_times, sperm_reps, run_details, net_stemcell_change, exploded, p01, p10, p12, p21, p23, p32 = \
            random_sampling(t01=t01, t10=t10, t12=t12, t21=t21, t23=t23, t32=t32, \
                           s0=s0, s1=s1, s2=s2, s3=s3, \
                           cycle_times=cycle_times, \
                           death_probabilities=death_probabilities, \
                           duplication_probabilities=duplication_probabilities, \
                           population_0=population_0, population_3=population_3, \
                           t_s=t_s, start_reps=start_reps)
        #print("hi 2")
        last_year, reps_r, counts_slope = score_prerun(sperm_times, sperm_reps, \
                                                       net_stemcell_change, run_details, exploded)
        #print("df length: " + str(len(results_df)))
        # add new line
        results_df.loc[len(results_df)] = p01, p10, p12, p21, p23, p32, last_year, reps_r, \
            counts_slope, net_stemcell_change, run_details, s0, s1, s2, s3, cycle_times, \
            death_probabilities, duplication_probabilities, population_0, population_3, \
            t_s, start_reps, t01, t10, t12, t21, t23, t32, death0, death1, death2, death3
    write_run_details_csv(csv_name, results_df)
    if one_death == True:
        # all deaths the same so only one included
        sm_df = results_df[['p01', 'p10', 'p12', 'p21', 'p23', 'p32', 'death0']]
    else:
        sm_df = results_df[['p01', 'p10', 'p12', 'p21', 'p23', 'p32', \
                               'death0', 'death1', 'death2', 'death3']]
    sm_df.to_csv(sm_file)
    
    return results_df


# earlier visualisations; see file "general_visualise_clean.py" for later version
def run_visualise_save(details_csv, time=0, reps=0, hexbin_name="temp_hex", \
                       counts_name="temp_counts", states_name="temp_states", do_log=True, \
                       t01="b", t10="b", t12="b", t21="b", t23="b", t32="b", \
                       s0="n", s1="n", s2="n", s3="c", \
                       cycle_times="16;16;16;16", population_0=100, population_3=1000, \
                       death_probabilities="0.01;0.01;0.01;0.01", \
                       duplication_probabilities="0.2;0.2;0.2;0.2", \
                       p01="0.8", p10="0.1", p12="0.8", p21="0.1", p23="0.8", p32="0.1", \
                       t_s=50, start_reps=30):
    
    # run with given set of parameters
    sperm_times, sperm_reps, run_details, net_stemcell_change, states_times_df, exploded = \
        population_run(t01=t01, t10=t10, t12=t12, t21=t21, t23=t23, t32=t32, \
                   s0=s0, s1=s1, s2=s2, s3=s3, \
                   cycle_times=cycle_times, \
                   death_probabilities=death_probabilities, \
                   duplication_probabilities=duplication_probabilities, \
                   p01=p01, p10=p10, p12=p12, p21=p21, p23=p23, p32=p32, \
                   population_0=population_0, population_3=population_3, \
                   t_s=t_s, start_reps=start_reps)
    
    last_year, reps_r, counts_slope = score_prerun(sperm_times, sperm_reps, \
                                                   net_stemcell_change, run_details, \
                                                   exploded)
            
    # make a dictionary of the set of arguments used 
    arguments_dict = assign_args_to_dict(t01, t10, t12, t21, t23, t32, \
                       s0, s1, s2, s3, \
                       cycle_times, death_probabilities, \
                       duplication_probabilities, \
                       p01, p10, p12, p21, p23, p32, \
                       population_0, population_3, t_s, start_reps)
    # add the results to the arguments dictionary
    arguments_dict.update({"last_year": last_year, "reps_r": reps_r, \
                           "counts_slope": counts_slope, "net_stemcell_change": net_stemcell_change})
    # convert to array for compatibility with Counter
    sperm_array_times = np.array(sperm_times)
    
    # plot reps against time (binned scatter plot due to high number of points)
    hex_fig = plt.figure()
    plt.hexbin(sperm_times, sperm_reps)
    plt.xlabel("time since puberty onset (days)")
    plt.ylabel("number of cell cycles (since puberty onset) before commitment")
    plt.show()
    hex_fig.savefig(hexbin_name + ".png")
    write_plot_details_csv(details_csv, hexbin_name + ".png", arguments_dict)
        
    # commited cell count against time
    # divide by 365 and round down for year
    sperm_array_years = np.floor((sperm_array_times + t_s) / 365)
    year_counts_dict = collections.Counter(sperm_array_years)
    scatter_fig = plt.figure()
    plt.scatter(year_counts_dict.keys(), year_counts_dict.values())
    plt.xlabel("year after puberty onset")
    plt.ylabel("number of commited cells")
    scatter_fig.show() 
    scatter_fig.savefig(counts_name + ".png")
    write_plot_details_csv(details_csv, counts_name + ".png", arguments_dict)

    #log count against time
    if do_log == True:
        log_counts_fig = plt.figure()
        plt.scatter(year_counts_dict.keys(), year_counts_dict.values())
        plt.yscale("log")
        plt.xlabel("year after puberty onset")
        plt.ylabel("number of commited cells")
        log_counts_fig.show() 
        log_counts_fig.savefig(counts_name + "_log.png")
        write_plot_details_csv(details_csv, counts_name + "_log.png", arguments_dict)
    
    
    # add a "years" column
    # log scale
    states_times_df["years"] = np.floor(states_times_df["time"] / 365)
    print(states_times_df.melt(id_vars=["time", "years"], value_vars=["state0", "state1", "state2", "state3"]))
    state_pop_graph = sns.relplot(states_times_df.melt(id_vars=["time", "years"], \
                                                       value_vars=["state0", "state1", "state2", "state3"]), \
                                  x="years", y="value", hue="variable", kind="line", palette="colorblind")
    state_pop_graph.set(yscale="log")
    state_pop_graph.savefig(states_name + ".png")
    write_plot_details_csv(details_csv, states_name + ".png", arguments_dict)
    print("reached end")

def plot_from_df(df, index, do_log, hexbin_name="temp_hex", counts_name="temp_counts", \
                 time=0, reps=0, population=10000):
    # use corresponding columns from dataframe 'line index' for each parameter
    # (rerun and plot for the set of parameters at index)
    run_visualise_save(time=time, reps=reps, population=population, hexbin_name=hexbin_name, \
                       counts_name=counts_name, do_log=do_log, \
                       t01=df.loc[index, 't01'], t10=df.loc[index, 't10'], t12=df.loc[index, 't12'], \
                       t21=df.loc[index, 't21'], t23=df.loc[index, 't23'], t32=df.loc[index, 't32'], \
                       s0=df.loc[index, 's0'], s1=df.loc[index, 's1'], \
                       s2=df.loc[index, 's2'], s3=df.loc[index, 's3'], \
                       cycle_times=df.loc[index, 'cycle_times'], \
                       death_probabilities=df.loc[index, 'death_probabilities'], \
                       duplication_probabilities=df.loc[index, 'duplication_probabilities'], \
                       p01=df.loc[index, 'p01'], p10=df.loc[index, 'p10'], p12=df.loc[index, 'p12'], \
                       p21=df.loc[index, 'p21'], p23=df.loc[index, 'p23'], p32=df.loc[index, 'p32'], \
                       t_s=df.loc[index, 't_s'], start_reps=df.loc[index, 'start_reps'])


# record information about saved plots
def write_plot_details_csv(csv_name, plot_file, arguments_dict):
    global figure_df
    arguments_dict["figure_name"] = plot_file
    try:
        figure_df = pd.read_csv(csv_name, index_col=0)
    # create a dataframe that will be later written as the file if it doesn't already exist
    except FileNotFoundError:
        figure_df = pd.DataFrame(columns=['figure_name', 't01', 't10', 't12', 't21', 't23', 't32', \
                                          'p01', 'p10', 'p12', 'p21', 'p23', 'p32', \
                                          's0', 's1', 's2', 's3', 'cycle_times', \
                                          'death_probabilities', 'duplication_probabilities', \
                                          'population_0', 'population_3', 't_s', 'start_reps', \
                                          'last_year', 'reps_r', 'counts_slope', 'net_stemcell_change'])
    # check for file name already in df
    # select row/s with particular value in a column (figure_name)
    figure_df.loc[figure_df["figure_name"] == plot_file]
    
    if any(figure_df["figure_name"] == plot_file):
        print("already present")
        figure_df.loc[figure_df["figure_name"] == plot_file] = [pd.Series(arguments_dict, \
                                                                         index=figure_df.columns)]
    else:
        figure_df.loc[len(figure_df)] = pd.Series(arguments_dict, index=figure_df.columns)
        print("not present")
    figure_df.to_csv(csv_name)
    
def write_run_details_csv(csv_name, results_df):
            
    # reorder columns to make sure they match
    runs_df = results_df[['t01', 't10', 't12', 't21', 't23', 't32', \
                        'p01', 'p10', 'p12', 'p21', 'p23', 'p32', \
                        's0', 's1', 's2', 's3', 'cycle_times', \
                        'death_probabilities', 'duplication_probabilities', \
                        'population_0', 'population_3', 't_s', 'start_reps', \
                        'last_year', 'reps_r', 'counts_slope', 'net_stemcell_change', \
                        'death0', 'death1', 'death2', 'death3']]
    runs_df.to_csv(csv_name, mode='w', index=False, header=True)

def assign_args_to_dict(t01="b", t10="b", t12="b", t21="b", t23="b", t32="b", \
                        s0="n", s1="n", s2="n", s3="c", \
                        cycle_times="16;16;16;16", \
                        death_probabilities="0.1;0.1;0.1;0.1", \
                        duplication_probabilities="0.2;0.2;0.2;0.2", \
                        p01="0.8", p10="0.1", p12="0.8", p21="0.1", p23="0.8", p32="0.1", \
                        population_0=100, population_3=1000, t_s=50, start_reps=30):
    arguments_dict = {
        "t01" : t01,
        "t10": t10,
        "t12": t12,
        "t21": t21,
        "t23": t23,
        "t32": t32,
        "s0": s0,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "cycle_times": cycle_times,
        "death_probabilities": death_probabilities,
        "duplication_probabilities": duplication_probabilities,
        "p01": p01,
        "p10": p10,
        "p12": p12,
        "p21": p21,
        "p23": p23,
        "p32": p32,
        "population_0": population_0,
        "population_3": population_3,
        "t_s": t_s,
        "start_reps": start_reps
    }
    return arguments_dict

# redundant; earlier version of parameter generation. generate_random_parameters_2
# used for final version
def generate_random_parameters(max_t_p=0.1, max_de_p=0.01, max_du_p=0.01, de_seperate0=False,
                               de_seperate1=False, de_seperate2=False, de_seperate3=False,
                               de_max0=False, de_max1=False, de_max2=False, de_max3=False,
                               du_seperate0=False, du_seperate1=False, du_seperate2=False, du_seperate3=False,
                               du_max0=False, du_max1=False, du_max2=False, du_max3=False):
    de_p_list = [np.random.uniform(0, max_de_p)] * 4
    # replace shared probability with new randomly generated probability
    # based on overall max_de_p, or on individual maxX if given
    de_seperate = [de_seperate0, de_seperate1, de_seperate2, de_seperate3]
    de_maxes = [de_max0, de_max1, de_max2, de_max3]
    for i in range(4):
        if de_seperate[i] == True:
            if de_maxes[i] == False:
                de_p_list[i] = np.random.uniform(0, max_de_p)
            else:
                de_p_list[i] = np.random.uniform(0, de_maxes[i])
    
    death_probabilities = ""
    for i in range(4):
        death_probabilities = death_probabilities + str(de_p_list[i])
        if i<3:
            death_probabilities = death_probabilities + ";"
    print(death_probabilities)
    
    t_ps = [0]*6
    for i in range(6):
        t_ps[i] = np.random.uniform(0, max_t_p)
    type(t_ps)
    
    du_p_list = [np.random.uniform(0, max_du_p)] * 4
    # replace shared probability with new randomly generated probability
    # based on overall max_du_p, or on individual maxX if given
    du_seperate = [du_seperate0, du_seperate1, du_seperate2, du_seperate3]
    du_maxes = [du_max0, du_max1, du_max2, du_max3]
    for i in range(4):
        if du_seperate[i] == True:
            if du_maxes[i] == False:
                du_p_list[i] = np.random.uniform(0, max_du_p)
            else:
                du_p_list[i] = np.random.uniform(0, du_maxes[i])
    
    duplication_probabilities = ""
    for i in range(4):
        duplication_probabilities = duplication_probabilities + str(du_p_list[i])
        if i<3:
            duplication_probabilities = duplication_probabilities + ";"
    print(duplication_probabilities)
    
    return duplication_probabilities, t_ps, duplication_probabilities


# give list of lists for which death rates/ duplication rates are paired together
# same number of elements in group_max as lists in grouping 
# default death rate lower for quiescent cells (not dividing so much less risk?)
# t_ps:
    # 0 : 0 -> 1
    # 1 : 1 -> 0
    # 2 : 1 -> 2
    # 3 : 2 -> 1
    # 4 : 2 -> 3
    # 5 : 3 -> 2
# default death probability grouping: quiescent (0 and 1) and active (2 and 3)
def generate_random_parameters_2(de_p_grouping=[[0,1],[2,3]], du_p_grouping=[[0,1,2,3]],
                                 t_p_grouping=[[0,1,2],[3,4,5]], group_max_t_p=[0.01,0.1],
                                 group_max_de_p=[0.01, 0.1], group_max_du_p=[0.1],
                                 group_min_t_p=[0,0], group_min_de_p=[0, 0], group_min_du_p=[0],):
    de_ps = [-2]*4
    du_ps = [-2]*4
    transition_probabilities = [-2]*6
    
    grouped_de_p = []
    # death probabilities
    for i in range(len(de_p_grouping)):
        p = str(np.random.uniform(group_min_de_p[i], group_max_de_p[i]))
        grouped_de_p.append(p)
        for j in de_p_grouping[i]:
            de_ps[j] = p
    death_probabilities = de_ps[0] + ";" + de_ps[1] + ";" + de_ps[2] + ";" + de_ps[3]
    
    grouped_du_p = []
    # duplication probabilities
    for i in range(len(du_p_grouping)):
        p = str(np.random.uniform(group_min_du_p[i], group_max_du_p[i]))
        grouped_du_p.append(p)
        for j in du_p_grouping[i]:
            du_ps[j] = p
    duplication_probabilities = du_ps[0] + ";" + du_ps[1] + ";" + du_ps[2] + ";" + du_ps[3]
    
    # transition probabilities
    grouped_t_p = []
    for i in range(len(t_p_grouping)):
        p = str(np.random.uniform(group_min_t_p[i], group_max_t_p[i]))
        grouped_t_p.append(p)
        for j in t_p_grouping[i]:
            transition_probabilities[j] = p
    return death_probabilities, duplication_probabilities, transition_probabilities, \
        grouped_de_p, grouped_du_p, grouped_t_p

# CLASSIFICATION FUNCTIONS
# Can add additional functions with different criteria by adding the 
# name of the new function to the dictionary 'score_function_dict',
# and using the name as the 'class_function' argument for 'generate_svm_input'

# Determines whether the run:
    # didn't expand excessively (exceed 100 * original population)
    # survived a minimum of 40 years
def score_run_minimal(sperm_times, sperm_reps, states_times_df, exploded):
    if exploded==False and sperm_times[-1] >= 14600:
        classification = 1
    else:
        classification = 0
    return classification

# Determines whether the run:
    # didn't expand excessively (exceed 100 * original population)
    # had a minimum combined population of 100 cells in states 2 and 3 at 40 years
def score_run_minimum_viable(sperm_times, sperm_reps, states_times_df, exploded):
    if exploded==True:
        return 0
    elif sperm_times[-1] < 14600:
        return 0
    # population of actively dividing SSCs (states 2 and 3) must be at least 100 at 40 years
    elif sum(states_times_df.iloc[int(np.ceil(14600/16)),3:5]) < 100:
        return 0
    else:
        return 1
    
score_function_dict = {
    "score_run_minimal": score_run_minimal,
    "score_run_minimum_viable": score_run_minimum_viable
    }

# order to store parameters in sm file:
    # transition probabilities (0 -> 1, 1 -> 0, 1 -> 2, 2 -> 1, 2 -> 3, 3 -> 2)
    # THEN any state behaviours
    # THEN death rate for each group
# headers saved in csv files to for later labelling of plots
# runs = number of sets of parameters to generate and run
def generate_svm_input(runs=1000, t01="b", t10="b", t12="b", t21="b", t23="b", t32="b",
                   s0="n", s1="n", s2="n", s3="c", cycle_times="16;16;16;16",
                   population_0=100, population_3=1000, t_s=50, start_reps=30, horizontal_cutoff=100,
                   sm_csv="default_sm.csv", ca_csv="default_ca.csv",
                   de_p_grouping=[[0,1],[2,3]], du_p_grouping=[[0,1,2,3]],
                   t_p_grouping=[[0,1,2],[3,4,5]], group_max_t_p=[0.01,0.1],
                   group_max_de_p=[0.01, 0.1], group_max_du_p=[0.1],
                   group_min_t_p=[0,0], group_min_de_p=[0, 0], group_min_du_p=[0],
                   class_function="score_run_minimal"):
    
    parameters = []
    
    transition_types = [t01, t10, t12, t21, t23, t32]
    transitions = ["01", "10", "12", "21", "23", "32"]
    
    # transition probabilities
    for i in range(len(t_p_grouping)):
        current_group_transitions = []
        for j in range(len(t_p_grouping[i])):
            if transition_types[t_p_grouping[i][j]] != "n":
                current_group_transitions.append(t_p_grouping[i][j])
        if len(current_group_transitions) > 1:
            name = "transition_p_" + "_".join([transitions[current_group_transitions[j]] \
                                               for j in range(len(current_group_transitions))])
            parameters.append(name)
        elif len(current_group_transitions) == 1:
            name = "transition_p_" + transitions[current_group_transitions[0]]
            parameters.append(name)

    # duplication probabilities
    state_behaviours = [s0, s1, s2, s3]
    for i in range(len(du_p_grouping)):
        current_group_duplicators = []
        for j in range(len(du_p_grouping[i])):
            if state_behaviours[du_p_grouping[i][j]] == "d":
                current_group_duplicators.append(du_p_grouping[i][j])
        if len(current_group_duplicators) > 1:
            name = "duplication_p_" + "_".join(str(k) for k in current_group_duplicators[i])
            parameters.append(name)
        elif len(current_group_duplicators) == 1:
            name = "duplication_p_" + str(current_group_duplicators[0])
            parameters.append(name)
    
    # death probabilities
    for i in range(len(de_p_grouping)):
        name = "death_p_" + "_".join(str(j) for j in de_p_grouping[i])
        parameters.append(name)
                
    sm_df = pd.DataFrame(columns=parameters)
    ca_df = pd.DataFrame(columns=["Class"])
    
    for i in range(runs):
        print(i)
        # generate parameters
        death_probabilities, duplication_probabilities, transition_probabilities, \
            grouped_de_p, grouped_du_p, grouped_t_p = \
                generate_random_parameters_2(de_p_grouping=de_p_grouping, du_p_grouping=du_p_grouping,
                                             t_p_grouping=t_p_grouping, group_max_t_p=group_max_t_p,
                                             group_max_de_p=group_max_de_p, group_max_du_p=group_max_du_p,
                                             group_min_t_p=group_min_t_p, group_min_de_p=group_min_de_p, 
                                             group_min_du_p=group_min_du_p,)
                
        # add parameters to sm dataframe
        all_parameters = grouped_t_p
        # only include duplication probabilities if at least one state is set to duplicate
        if s0=="d" or s1=="d" or s2=="d" or s3=="d":
            all_parameters.extend(grouped_du_p)
        all_parameters.extend(grouped_de_p)
        
        # next line in sm_df
        sm_df.loc[len(sm_df)] = all_parameters
        
        # run model with specified structure and generated parameters
        sperm_times, sperm_reps, run_details, net_stemcell_change, states_times_df, exploded = \
            population_run(t01=t01, t10=t10, t12=t12, t21=t21, t23=t23, t32=t32, \
                           s0=s0, s1=s1, s2=s2, s3=s3, \
                           cycle_times=cycle_times, \
                           death_probabilities=death_probabilities, \
                           duplication_probabilities=duplication_probabilities, \
                           p01=transition_probabilities[0], \
                           p10=transition_probabilities[1], \
                           p12=transition_probabilities[2], \
                           p21=transition_probabilities[3], \
                           p23=transition_probabilities[4], \
                           p32=transition_probabilities[5], \
                           population_0=population_0, population_3=population_3, \
                           t_s=t_s, start_reps=start_reps, horizontal_cutoff=horizontal_cutoff, \
                           write_to_csv=False)
        
        # evaluate whether the model 'succeeded'
        # classified according to the criteria of the model given for the 
        # 'class_function' argument
        classification = \
            score_function_dict[class_function](sperm_times=sperm_times, sperm_reps=sperm_reps, \
                                                states_times_df=states_times_df, exploded=exploded)
        ca_df.loc[len(ca_df)] = classification
        
    # write sm and ca dataframes to csv files
    sm_df.to_csv(sm_csv, header=True, index=False)
    ca_df.to_csv(ca_csv, header=True, index=False)
    print(sum(ca_df["Class"]))


# generate a set of 12 plots of state populations against time, with a randomly generated
# set of parameters for each plot
def visualise_random_set(t01="b", t10="b", t12="b", t21="b", t23="b", t32="b",
                         s0="n", s1="n", s2="d", s3="c", cycle_times="16;16;16;16",
                         population_0=100, population_3=1000, t_s=50, start_reps=30, horizontal_cutoff=100,
                         de_p_grouping=[[0,1],[2,3]], du_p_grouping=[[0,1,2,3]],
                         t_p_grouping=[[0,1,2],[3,4,5]], group_max_t_p=[0.01,0.1],
                         group_max_de_p=[0.01, 0.1], group_max_du_p=[0.1],
                         group_min_t_p=[0,0], group_min_de_p=[0, 0], group_min_du_p=[0],
                         fix_t_p = False, log_scale=True, do_save=True, 
                         save_fig="12_state_plots_default.png"):
    
    fig, axs = plt.subplots(nrows=3, ncols=4, sharex=True, figsize=(20,13))
    
    parameters = []
    transition_types = [t01, t10, t12, t21, t23, t32]
    transitions = ["01", "10", "12", "21", "23", "32"]
    
    # transition probabilities
    # only done if transition probabilites aren't fixed
    if fix_t_p==False:
        for i in range(len(t_p_grouping)):
            current_group_transitions = []
            for j in range(len(t_p_grouping[i])):
                if transition_types[t_p_grouping[i][j]] != "n":
                    current_group_transitions.append(t_p_grouping[i][j])
            if len(current_group_transitions) > 1:
                name = "transition_p_" + "_".join([transitions[current_group_transitions[j]] \
                                                   for j in range(len(current_group_transitions))])
                parameters.append(name)
            elif len(current_group_transitions) == 1:
                name = "transition_p_" + transitions[current_group_transitions[0]]
                parameters.append(name)
    
    # duplication probabilities
    state_behaviours = [s0, s1, s2, s3]
    for i in range(len(du_p_grouping)):
        current_group_duplicators = []
        for j in range(len(du_p_grouping[i])):
            if state_behaviours[du_p_grouping[i][j]] == "d":
                current_group_duplicators.append(du_p_grouping[i][j])
        if len(current_group_duplicators) > 1:
            name = "duplication_p_" + "_".join(str(k) for k in current_group_duplicators[i])
            parameters.append(name)
        elif len(current_group_duplicators) == 1:
            name = "duplication_p_" + str(current_group_duplicators[0])
            parameters.append(name)
    
    # death probabilities
    for i in range(len(de_p_grouping)):
        name = "death_p_" + "_".join(str(j) for j in de_p_grouping[i])
        parameters.append(name)
    
    #print("reached run")
    for i in range(12):
        print(i)
        death_probabilities, duplication_probabilities, transition_probabilities, \
            grouped_de_p, grouped_du_p, grouped_t_p = \
                generate_random_parameters_2(de_p_grouping=de_p_grouping, du_p_grouping=du_p_grouping,
                                             t_p_grouping=t_p_grouping, group_max_t_p=group_max_t_p,
                                             group_max_de_p=group_max_de_p, group_max_du_p=group_max_du_p,
                                             group_min_t_p=group_min_t_p, group_min_de_p=group_min_de_p, 
                                             group_min_du_p=group_min_du_p,)
        # if transition probabilites are fixed, use the argument instead of the randomly generated values
        if fix_t_p != False:
            transition_probabilities = fix_t_p
        sperm_times, sperm_reps, run_details, net_stemcell_change, states_times_df, exploded = \
            population_run(t01=t01, t10=t10, t12=t12, t21=t21, t23=t23, t32=t32, \
                           s0=s0, s1=s1, s2=s2, s3=s3, \
                           cycle_times=cycle_times, \
                           death_probabilities=death_probabilities, \
                           duplication_probabilities=duplication_probabilities, \
                           p01=transition_probabilities[0], \
                           p10=transition_probabilities[1], \
                           p12=transition_probabilities[2], \
                           p21=transition_probabilities[3], \
                           p23=transition_probabilities[4], \
                           p32=transition_probabilities[5], \
                           population_0=population_0, population_3=population_3, \
                           t_s=t_s, start_reps=start_reps, horizontal_cutoff=horizontal_cutoff, \
                           write_to_csv=False)
        if fix_t_p == False:
            parameters_values = grouped_t_p
            parameters_values.extend(grouped_du_p)
        # if transition probabilities are fixed they are not included as parameters
        else:
            parameters_values = grouped_du_p
        parameters_values.extend(grouped_de_p)

        states_times_df["year"] = np.floor(np.array(states_times_df["time"])/365)
        melted_states_df = states_times_df.melt(id_vars=["time", "year"], value_vars=["state0", "state1", "state2", "state3"])
        if i!=11:
            sns.lineplot(data=melted_states_df, x="year", y="value", hue="variable", \
                         ax=axs[int(np.floor(i/4)),i%4], errorbar=None, legend=None)
        else:
            sns.lineplot(data=melted_states_df, x="year", y="value", hue="variable", \
                         ax=axs[int(np.floor(i/4)),i%4], errorbar=None)
        axs[int(np.floor(i/4)),i%4].set_xlabel("")
        axs[int(np.floor(i/4)),i%4].set_ylabel("")
        axs[int(np.floor(i/4)),i%4].set_title("\n".join(textwrap.wrap("; ".join([str(round(float(i),4)) for i in parameters_values]), width=40)), fontsize=9)
        if log_scale==True:
            axs[int(np.floor(i/4)),i%4].set(yscale="log")
    plt.legend(loc="center left", bbox_to_anchor=(1,1.5))
    fig.supxlabel("Time (years)", x=0.5, y=0.04, fontsize=25)
    fig.supylabel("Average population each year", x=0.07, y=0.5, fontsize=25)
    fig.text(0.5, 0.005, "\n".join(textwrap.wrap("Parameters: " + "; ".join(parameters), width=160)), ha="center", va="bottom", fontsize=15)
    if do_save==True:
        plt.savefig(save_fig, bbox_inches="tight", pad_inches=0.1)

