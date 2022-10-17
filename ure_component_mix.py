""" This is a script to generate a dataset of hydrocarbon mixtures composed of the components selected by Dr. Andrew Ure
 in his ACS presentation slides."""

# import pandas as pd
# import os
import model_func
import numpy as np
# from sklearn import linear_model as lin_mod
# from sklearn.model_selection import train_test_split

# In order to generate all possible compositions we will treat it as a partitioning exercise.
# 100% needs to be partitioned between 7 components(y). 100% can be represented as x objects corresponding to (100/x)%.
# The number of possible partitions is known from theory of combinatorics.
# There are two formulae we could use:
#       If concentration = 0 is allowed         --->        n_compositions = nCr(x+(y-1), y-1)
#       If concentration = 0 is forbidden       --->        n_compositions = nCr(x-1, y-1)
# This is commonly visualised using the stars and bars convention, where stars are objects and bars are partitions.
############### Stars and bars example ###############
# There are 4 objects (*) to be partitioned (|) into 3 groups.
# Empty groups are allowed.
# This can be done in nCr(4+(3-1), 3-1) = 15 ways.
# 1. |****||| 2. ||****|| 3. |||****| 4. |***|*|| 5. |***||*| 6. |*|***|| 7. ||***|*| 8. |*||***| 9. ||*|***|
# 10. |**|*|*| 11. |*|**|*| 12. |*|*|**| 13. |**|**|| 14. |**||**| 15. ||**|**|
#
# Empty groups are forbidden.
# This can be done in nCr(4-1, 3-1) = 3 ways.
# 1. |**|*|*| 2. |*|**|*| 3. |*|*|**|

############### Rationale for formulation strategy ###############
# Two things will be done with the dataset generated here:
# First, the model reported by A. Ure will be tested against this dataset, constructed from the same components listed
# in his published method, and hopefully representative of that dataset.
# Second, the data will be used to train a new model using the same method described by A. Ure and differences between
# these models will be identified, evaluated, and an assessment made as to their cause.
#
# The models were reported to have been trained and validated on two sets of 20,000 unique mixtures each. We therefore
# want to choose a concentration increment which will allow at least 40,000 mixtures to be generated.
# As these mixtures are intended to be representative of fossil jet fuels, it would seem logical that no component
# should ever be entirely absent from a mixture since each component represents a class of components which would not
# be expected to be absent in fossil jet fuel. Therefor, we should use the second formula above when deciding on a
# suitable concentration increment.

# Here we calculate the concentration increment for every integer number of steps counting down from 100.
# for n_steps in range(100,0,-1):
#     # Print results to the console for possible future reference or reporting.
#     print("There are: ", model_func.composition_count(n_steps, 7),
#           "possible compositions using {} x{}% increments".format(n_steps, round(100.0/n_steps,2)))
#     if model_func.composition_count(n_steps, 7)<40000:
#         # When an increment is found that results in fewer than 40,000 unique mixtures print out the previous increment.
#         print()
#         print("{}% is the largest mol% increment to give more than 40,000 unique mixtures.\nThis gives {} equal"
#               " concentration steps.".format(round(100/(n_steps+1), 3), n_steps+1))
#         print()
#         step_count = n_steps + 1
#         break  # Break the loop when the target has been found.

# There are just over 54,000 possible compositions using a ~4.545% increment.

############### Generating the mixtures ###############
# There's probably an easier way to read files in python. I will find out what that is at some point.
component_file = 'Andrews_components.xlsx'

components = pd.read_excel(component_file)  # Read in descriptor data for the components.
print(components.to_string())  # Print the data to check if it read in correctly.

concs = model_func.mix_concs(step_count, 7)  # Create a list of all possible mixture concentrations.

# Random Concentrations
# The above approach produced a model with errors heavily affected by the concentration of high viscosity components.
# An alternative approach using randomly generated concentrations was adopted instead.
concs = model_func.rand_concs(7, 40000)

# Print out a few concentration arrays and their sums to make sure they add up to 1.0.
for i in range(10):
    print(concs[i], "----", np.sum(concs[i]))

# Combine the array of concentrations with the component atom types and viscosities to generate the full dataset.
temperatures = [273.15, 283.15, 293.15, 303.15, 308.15, 313.15]
# Functions were easier to write when a list of temperatures was provided. These were used by A. Ure.
[dataset, andrades] = model_func.make_dataset(concs, components, temperatures)
# dataset.to_excel("A.Ure_replica_dataset.xlsx")
dataset.to_excel("A_Ure_replica_dataset_random_v4.xlsx")
# andrades.to_excel("Andrade_Parameters.xlsx")
