import numpy as np
from pyssem.model import Model
import pandas as pd
import matplotlib.pyplot as plt
import json

class YearDTO:

    def __init__(self, x0):
        self.tax_revenue_lastyr = 0
        self.shell_revenue = None
        self.removals_left = 0
        self.old_environment = x0
        self.money_bucket_1 = 0
        self.money_bucket_2 = 0
        self.opt_tax_revenue = 0
        self.opt_shell_revenue = None
        self.opt_environment = None
        self.opt_money_1 = 0
        self.opt_money_2 = 0
        self.opt_removals = 0


    def store_year_data(self, tax_revenue_lastyr, shell_revenue, removals_left, current_environment, money_bucket_1, money_bucket_2):
        self.tax_revenue_lastyr = tax_revenue_lastyr
        self.shell_revenue = shell_revenue
        self.removals_left = removals_left
        self.old_environment = current_environment
        self.money_bucket_1 = money_bucket_1
        self.money_bucket_2 = money_bucket_2

    def update_year_data(self, tax_revenue_lastyr, shell_revenue, removals_left, current_environment, money_bucket_1, money_bucket_2):
        self.opt_tax_revenue = tax_revenue_lastyr
        self.opt_shell_revenue = shell_revenue
        self.opt_money_1 = money_bucket_1
        self.opt_money_2 = money_bucket_2
        self.opt_removals = removals_left
        self.opt_environment = current_environment
