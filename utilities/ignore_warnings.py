"""
Created by Christos Baziotis.
"""
import warnings


# todo:remove DeprecationWarning
def set_ignores():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
