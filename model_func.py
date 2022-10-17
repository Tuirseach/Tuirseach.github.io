"""This is a repository for the functions to be used in the viscosity modelling work."""


def composition_count(step_count, component_count):
    from math import comb
    n = step_count - component_count
    k = component_count
    n_compositions = comb(n+(k-1), n)
    return n_compositions


def mix_concs(step_count, component_count, total=1):
    from itertools import combinations
    from numpy import diff
    inc_val = total/step_count  # Determine the increment value
    # Referring back to the stars and bars analogy, if there are n increments (stars) to be partitioned between k
    # components, there are k-1 bars to be placed in n-1 spaces between increments. Enumerating these spaces allows for
    # generation of every possible partition index.
    # The difference between each partition index is equal to the number of increments for that component. Multiplying
    # the number of increments by the increment value returns the mol% of the component in the mixture.
    concentrations = [diff([0]+list(comp)+[step_count])*inc_val for comp in combinations(range(1, step_count),
                                                                                         component_count-1)]
    return concentrations


def viscosity_at_temp(B, T0, T=243.15):
    from math import exp
    viscosity = exp(B*((1/T)-(1/T0)))
    return viscosity


def mix_viscosity(viscosities, concentrations):
    contributions = [v**3 for v in viscosities]*concentrations
    visc_out = contributions.sum()**(1/3)

    return visc_out


def get_andrade(viscosities, temperatures):
    from scipy import stats
    import math
    slope, intercept = stats.linregress(x=[1 / t for t in temperatures], y=[math.log(v) for v in viscosities])[:2]
    B = slope
    T0 = -slope/intercept
    return B, T0


def make_dataset(concentrations, components, temperatures, target=243.15,
                 a_types=('pCH3', 'pCH2', 'pCH', 'pC', 'cyCH2', 'cyCH', 'alCH3', 'alCH2', 'alCH', 'ArCH', 'ArC',
                          'rjC')):
    import pandas as pd
    import numpy as np
    # Create a pandas dataframe with the concentrations of each component.
    dataset = pd.DataFrame(concentrations, columns=components['Component'])
    dataset.columns.names = ['Mixture']
    andrades = pd.DataFrame()
    # Create empty lists to be filled with mixture descriptors.
    mw = []
    a_fractions = []
    viscosities = []
    b_all = []
    t0_all =[]
    # Loop through each concentration and fill the descriptor lists.
    for row in concentrations:
        # The molecular weight is the dot product of arrays containing the component concentration and
        # their molecular weight
        mw.append(np.dot(components['MW'].values.tolist(), row))
        atoms = []
        # Loop through each atom type
        for a_type in a_types:
            atoms.append(np.dot(components[a_type].values.tolist(), row))
        atoms = np.array(atoms)
        a_fractions.append(atoms/atoms.sum())
        v_heads = ['Viscosity @ {}'.format(t) for t in temperatures]
        v_mix = []
        for head in v_heads:
            v_mix.append(mix_viscosity(components[head].values, row))
        B, T0 = get_andrade(v_mix, temperatures)
        viscosities.append(viscosity_at_temp(B, T0))
    dataset['MW'] = mw
    dataset[list(a_types)] = a_fractions
    dataset['Viscosity @ 243.15'] = viscosities

    return dataset

def rand_concs(n_comps, n_concs):
    """Generate a set of n_concs compositions for n_comps components."""
    from numpy.random import randint
    concs = randint(1, 1000, [n_concs, n_comps])
    concs = concs/concs.sum(axis=1)[:, None]
    return concs

def coefs_and_metrics(file):
    """This function takes a file containing a models coefficients and metrics and prints them to the console."""
    import pandas as pd
    data = pd.read_excel(file)
    for column in data.columns:
        print("{} = {}".format(column, round(data[column][0], 3)))
