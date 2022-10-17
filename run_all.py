"""This script is used to parse through the main modelling process in a single run."""
import os

exec(open('ure_component_mix.py').read())
exec(open('replica_model.py').read())
exec(open('replica_model_visualise.py').read())
