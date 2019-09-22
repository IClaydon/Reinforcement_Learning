import datetime as dt
from calendar import monthrange
import csv
import attr
import numpy as np
import pandas as pd

from person import Person, BOOKING_COLUMN_NAMES


def booking_lead_time_of_customer_universe(gamma, lead_time):
    alpha = gamma
    # Worth adding a drop towards 0 rate at the end, but left for later
    return np.power(alpha, 2) / (np.power(alpha, 2) + np.power(lead_time, 2))

def lead_days_distribution(gamma, max_lead_days):
    result = [booking_lead_time_of_customer_universe(gamma, lead_time)
                              for lead_time in range(max_lead_days+1)]
    return result



@attr.s
class Market(object):
    start_date=attr.ib()
    start_reserved_night_date = attr.ib()
    end_reserved_night_date = attr.ib()
    average_number_of_potential_customers = attr.ib()
    capacity = attr.ib()
    minimum_budget_in_population = attr.ib(default=40)
    gamma = attr.ib(default=10)
    gamma_weights_months = attr.ib(factory=list)
    gamma_weights_dow=attr.ib(factory=list)
    month_prob = attr.ib(factory=list)
    dow_prob = attr.ib(factory=list)

    _budgets = attr.ib(default=None)
    _first_start_dates = attr.ib(default=None)
    _number_of_potential_customers_per_month = attr.ib(default=None)


    #@profile
    def _initialize_budgets(self, total_potential_customers):
	#### Take a normal distribution, truncated at minimum price.
        budgets = np.abs(np.random.normal(0,75,total_potential_customers))+self.minimum_budget_in_population

        if len(budgets) != total_potential_customers:
            raise Exception
        return budgets
      


    #@profile
    def generate_bookings_for_lead_time(self,revenue_manager,person_np,time,where):
        bookings_df = None


        person_np_book= person_np[where]

        for i in range(len(person_np_book)):
            a = Person(budget=person_np_book[i,0],possible_start_dates=person_np_book[i,1],     possible_end_dates=person_np_book[i,2],possible_booking_lead_times=person_np_book[i,3],type_of_person='transient')
            rates_table_np = revenue_manager.get_rates_table_for_reserved_night_date(time,person_np_book[i,1][0])
            temp_bookings_df = a.generate_bookings(time,rates_table_np)
            if temp_bookings_df is None:
                continue
            if bookings_df is None:
                bookings_df = temp_bookings_df
            else:
                bookings_df = np.append(bookings_df,temp_bookings_df,axis=0)
        if bookings_df is None:
            return revenue_manager
        revenue_manager.add_bookings(bookings_df)
        return revenue_manager


    #@profile
    def _create_persons(self, reserved_night_date,pers_writer):

        day = reserved_night_date.weekday()
        month = reserved_night_date.month-1
        total_potential_customers = self.average_number_of_potential_customers *self.month_prob[month]*self.dow_prob[day]
        self._budgets = self._initialize_budgets(int(total_potential_customers))
        budgets = self._budgets
        

        gamma = self.gamma*self.gamma_weights_months[reserved_night_date.month-1]*self.gamma_weights_dow[reserved_night_date.weekday()]
        
       	lead_days_dist = lead_days_distribution(gamma, int(gamma * 8))


        for i in range(len(budgets)):
            
            x = np.arange(0,int(gamma*8)+1)
            
            lead_days_to_use = np.random.choice(x,
                                                p=lead_days_dist / np.sum(lead_days_dist))
            lead_days_to_use = int(str(lead_days_to_use))
            length_of_stay = 1 # for now we just do an LOS of 1
            budget = budgets[i]
            
            start_dates = [reserved_night_date]


            possible_start_dates=np.array(start_dates)
            possible_end_dates=np.array([start_date+dt.timedelta(days=length_of_stay) for start_date in start_dates])
            possible_booking_lead_times=np.array([lead_days_to_use])

            lead_dates = np.array([start_date - dt.timedelta(days=lead_days_to_use) for start_date in start_dates])
            days_since =(start_dates[0]-dt.timedelta(days=lead_days_to_use) - self.start_date).days
            new_person = np.array([budget,possible_start_dates,possible_end_dates,possible_booking_lead_times,lead_dates,'transient',days_since])
            
            if i ==0:
                person_np = new_person
            else:
                person_np=np.vstack([person_np,new_person])

            pers_writer.writerow([lead_days_to_use,budget])
        return person_np

 


