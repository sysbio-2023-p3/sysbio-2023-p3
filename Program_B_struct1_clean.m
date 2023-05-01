% no s4 argument because there's no initial population for state 4 (commited)
% no r43 because state 4 are commited and can't return to earlier states
% r =  rate per day per cell
% maxTime = maximum time to follow
% dX : death rate for each state
% write_output : true if output should be written to a file
% write_file : name of file to write output to
function [x, times] = gillespie5(s0, s1, s2, s3, r01, r10, r12, r21, r23, r32, r34, ...
    d0, d1, d2, d3, maxTime, write_output, write_file)
    % very rough upper bound estimate of number of time steps to avoid
    % needing to append to an array
    approx_array = ceil((max([r01, r10, r12, r21, r23, r32, r34]) + max([d0, d1, d2, d3])) * ...
        (s0 + s1 + s2 + s3) * maxTime);
    x = zeros(5, approx_array);
    % populations in each state
    x(:,1) = [s0; s1; s2; s3; 0];
    % only has simple transitions, transposed so that the changes for each
    % state in a given event could be written as a row for readability
    % order of event rows:
        % transitions (in the order they are given as arguments)
        % commitment (asymmetrical division 3 -> 3 + 4)
        % death (states 0-4)
    events = transpose([-1, 1, 0, 0, 0; ...
        1, -1, 0, 0, 0; ...
        0, -1, 1, 0, 0; ...
        0, 1, -1, 0, 0; ...
        0, 0, -1, 1, 0; ...
        0, 0, 1, -1, 0; ...
        0, 0, 0, 0, 1; ...
        -1, 0, 0, 0, 0; ...
        0, -1, 0, 0, 0; ...
        0, 0, -1, 0, 0; ...
        0, 0, 0, -1, 0]);
    
    i = 2;
    timeSteps = zeros(1, approx_array);
    times = zeros(1,approx_array);
    % exclude row 5 (commited cells) as this is a running count not subject
    % to death
    % check that the total population of states 0-3 (non-committed cells)
    % has not fallen to 0
    while  times(i-1) < maxTime && sum(x(1:4,i-1)) > 0
        rates = [r01*x(1,i-1), r10*x(2,i-1), r12*x(2,i-1), r21*x(3,i-1), r23*x(3,i-1), r32*x(4,i-1), r34*x(4,i-1), ...
            d0*x(1,i-1), d1*x(2,i-1), d2*x(3,i-1), d3*x(4,i-1)];
        totalRate = sum(rates);
        eventProbs = rates / totalRate;
        currentStep = -log(1-rand)/totalRate;
        timeSteps(i-1) = currentStep;
        times(i) = (times(i-1) + currentStep);
        pE = rand;
        currentEvent = sum(pE > cumsum(eventProbs)) + 1;
        x(:,i) = x(:,i-1) + events(:, currentEvent);
        i = i + 1;
    end
    if write_output == true
        % add times as a column in x
        x(6,:) = times;
        % exclude rows where all values are 0
        % any is true if any value is non 0
        x = transpose(x);
        x = x(any(x,2),:);
        writematrix(x, write_file);
    end
end


% return 'classifications', list of 1s (survived at least x years) or 0s
% (died out within x years), and 'samples', matrix with a row for each sample
% order of columns in samples; p01, p10, p12, p21, p23, p32, d0, d1, d2, d3
% rates generated in days (readability) then converted to per day
% eg maxDeath = expected number of days before death
% pale_cycle; a fixed value, data based rather than exploratory
% one_death: if true, only generate one death rate and use for all states
% include file extensions in file name arguments
function [classifications, samples] = run_randomised(maxDeath, minDeath, N, pale_cycle, ...
    one_death, max_trans_len, min_trans_len, s0, s1, s2, s3, maxTime, ...
    min_survival, samples_file, classifications_file)
    classifications = zeros(1,N);
    if one_death == true
        death_ts = randi([minDeath, maxDeath], [N,1]);
        trans_ts = randi([min_trans_len, max_trans_len], [N,6]);
        samples = [trans_ts death_ts];
        size(samples)
        for i = 1:N
            disp(i)
            [x, times] = gillespie5(s0, s1, s2, s3, 1/trans_ts(i,1), ...
                1/trans_ts(i,2), 1/trans_ts(i,3), 1/trans_ts(i,4), ...
                1/trans_ts(i,5), 1/trans_ts(i,6), 1/pale_cycle, ...
                1/death_ts(i), 1/death_ts(i), 1/death_ts(i), 1/death_ts(i), ...
                maxTime, false, "na");
            % the first occurance of the maximum value of state 4
            % 'population' is the end of sperm production
            [M, I] = max(x(5,:));
            oldest_sperm = times(I);
            if oldest_sperm >= min_survival
                classifications(i) = 1;
            end
        end
    else
        death_ts = randi([minDeath, maxDeath], [N,4]);
        trans_ts = randi([min_trans_len, max_trans_len], [N,6]);
        % parameters include a death rate for each state
        samples = [trans_ts death_ts];
        size(samples);
        for i = 1:N
            [x, times] = gillespie5(s0, s1, s2, s3, 1/trans_ts(i,1), ...
                1/trans_ts(i,2), 1/trans_ts(i,3), 1/trans_ts(i,4), ...
                1/trans_ts(i,5), 1/trans_ts(i,6), 1/pale_cycle, ...
                1/death_ts(i,1), 1/death_ts(i,2), 1/death_ts(i,3), ...
                1/death_ts(i,4), maxTime, false, "na");
            [M, I] = max(x(5,:));
            oldest_sperm = times(I);
            if oldest_sperm >= min_survival
                classifications(i) = 1;
            end
        end
    end
    % convert to tables with headers for compatibility with plotting and
    % svm training
    if one_death == true
        samples_table = array2table(samples, "VariableNames", ...
            ["t01", "t10", "t12", "t21", "t23", "t32", "d_all"]);
    else
        samples_table = array2table(samples, "VariableNames", ...
            ["t01", "t10", "t12", "t21", "t23", "t32", "d0", ...
            "d1", "d2", "d3"]);
    end
    classifications_table = array2table(transpose(classifications), "VariableNames", ["Class"]);
    writetable(classifications_table, classifications_file);
    writetable(samples_table, samples_file);

end








