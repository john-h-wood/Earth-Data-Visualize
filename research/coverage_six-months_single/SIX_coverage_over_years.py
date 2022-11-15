import edcl as di
import numpy as np
import matplotlib.pyplot as plt

# Constants
MINIMUM_COVERAGES = (0.99,)

# This script plots the frequency of majority coverage over years
relative_points = di.load_pickle('/Volumes/LaCie/SURF/Code/data_vis_6/pickles/relative_points.pickle')
time_stamps = di.load_pickle('/Volumes/LaCie/SURF/Code/data_vis_6/pickles/time_stamps.pickle')

# Initialize plot
plt.figure(figsize=(12, 8), dpi=300)
plt.title(f'Relative Occurrences of $\geq${di.iterable_to_words(MINIMUM_COVERAGES)} Coverage over Years')
plt.xlabel('Year')
plt.ylabel('Relative occurrences')

# Plot coverage occurrences over years for each minimum coverage
for minimum_coverage in MINIMUM_COVERAGES:
    # Find time stamps with minimum to 1 coverage
    high_indices = np.asarray((minimum_coverage <= relative_points) & (relative_points <= 1)).nonzero()[0]

    high_times = list()
    for i in high_indices:
        high_times.append(time_stamps[i])

    # Count number of occurrences for each unique year
    years = list()
    occurrences = list()

    for time in high_times:
        year = di.get_winter_year(time)
        if year in years:  # year has already been found, so increase occurrences for that year by 1
            occurrences[years.index(year)] += 1
        else:  # new year, add it to years and a new counter to occurrences starting at 1
            years.append(year)
            occurrences.append(1)

    # Convert lists of years and occurrences to numpy arrays
    years = np.array(years)
    occurrences = np.array(occurrences)

    # Normalize occurrences
    occurrences = occurrences / len(high_times)

    plt.plot(years, occurrences, '--o', label=f'$\geq${minimum_coverage}')

# Add legend to and save plot
plt.legend()
plt.savefig('relative_occurrences_over_years.png')
