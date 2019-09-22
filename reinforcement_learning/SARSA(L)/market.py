import datetime as dt
from calendar import monthrange
import csv
import attr
import numpy as np
import pandas as pd

from person import Person, BOOKING_COLUMN_NAMES

### Market.py for Forward View N-step SARSA(lambda) RL algorithm


def booking_lead_time_of_customer_universe(gamma, lead_time):
    alpha = gamma
    
    lambda_max = 112 # number of rooms
    normalization = 2 / np.pi
    # Worth adding a drop towards 0 rate at the end, but left for later
    return normalization * alpha * lambda_max / (np.power(alpha, 2) + np.power(lead_time, 2))

def lead_days_distribution(gamma, max_lead_days):
    result = [booking_lead_time_of_customer_universe(gamma, lead_time)
                              for lead_time in range(max_lead_days+1)]
    return result



def probability_day_of_week(weekday):
    #return 1/7
    weekday_probability = [1,1,1,1,1,1,0.7]
    return weekday_probability[weekday]/np.sum(weekday_probability)
    """if weekday in [0,1,2,3,4, 5]:
        return weekday_probability
    else:
        return 0.8"""


def month_probability(month):
    # from demo merchant, for now
    """month_probabilities = {
        1: 0.2,
        2: 0.05,
        3: 0.1,
        4: 0.15,
        5: 0.2,
        6: 0.35,
        7: 0.3,
        8: 0.25,
        9: 0.25,
        10: 0.2,
        11: 0.15,
        12: 0.1,
    }"""
    month_probabilities = {
        1: 0.8,
        2: 0.9,
        3: 1.,
        4: 0.9,
        5: 1.,
        6: 0.95,
        7: 0.95,
        8: 1.0,
        9: 1.0,
        10: 1.0,
        11: 1.1,
        12: 1.2,
    }
    return month_probabilities[month] / 11.8


@attr.s
class Market(object):
    start_reserved_night_date = attr.ib()
    end_reserved_night_date = attr.ib()
    average_number_of_potential_customers = attr.ib()
    capacity = attr.ib()
    state=attr.ib()
    value_DOW=attr.ib()
    value_month=attr.ib()
    min_budget = attr.ib(default=40)
    gamma = attr.ib(default=10)
    gamma_weights_months = attr.ib(factory=list)
    gamma_weights_dow=attr.ib(factory=list)
    month_prob = attr.ib(factory=list)
    dow_prob = attr.ib(factory=list)

    _budgets = attr.ib(default=None)
    _first_start_dates = attr.ib(default=None)
    _number_of_potential_customers_per_month = attr.ib(default=None)

    #@profile
    def initialize_market(self):
	
	#### Define number of customers
        total_potential_customers = self.average_number_of_potential_customers * \
                                    (self.end_reserved_night_date - self.start_reserved_night_date).days
        
	#### Define budget of each customer
        self._budgets = self._initialize_budgets(total_potential_customers)

	#### Define when customer wants to book (day,month,year)
        cumulative_sum = np.cumsum([month_probability(i) for i in range(1, 13)])
        months = [self._calculate_month(uniform,cumulative_sum) for uniform in np.random.uniform(0, 1, total_potential_customers)]
        years = [self._calculate_year(uniform) for uniform in np.random.uniform(0, 1, total_potential_customers)]

        cumulative_sum_weekday = np.cumsum([probability_day_of_week(i) for i in range(7)])
        self._first_start_dates = [dt.datetime(year, month, self._calculate_day(year, month,cumulative_sum_weekday))
                                   for year, month in zip(years, months)]
	### Define how many customers per month
        self._fill_generation_statistics()

    #@profile
    def _initialize_budgets(self, total_potential_customers):
	#### Take a normal distribution, truncated at minimum price.
        budgets = np.abs(np.random.normal(0,75,total_potential_customers))+self.min_budget

        #budgets = budgets[budgets > self.minimum_budget_in_population]
        #budgets = budgets[:total_potential_customers]
        if len(budgets) != total_potential_customers:
            raise Exception
        return budgets
    
    
    #@profile
    def generate_bookings_for_reserved_night_date(self, reserved_night_date, revenue_manager, person_np,run):
        
        start_date = reserved_night_date  - dt.timedelta(days=int(self.gamma * 8))
        end_date = reserved_night_date + dt.timedelta(days=1)
        day = reserved_night_date.weekday()
        month=reserved_night_date.month-1
        #print(person_np[:,4]-start_date,person_np[:,6])
        N_bookings=0
        rev=0
        wheres = np.array([])
        each_step = np.array([])
        for extra_days in range((end_date-start_date).days):
            time = start_date + dt.timedelta(days=extra_days)
            rates_table_np = revenue_manager.get_rates_table_for_reserved_night_date(time, reserved_night_date,self.state,N_bookings,self.gamma*8-extra_days,run,self.value_DOW,self.value_month)

            #print(person_np[:,4]-start_date)
            bookings_df = self.generate_bookings(time,rates_table_np,person_np,int((reserved_night_date-time).days))

            if bookings_df is None:
                
                if len(rates_table_np)>0:
                    
                    price=rates_table_np[0][3]
                    arguement = (((self.gamma*8)-extra_days)*self.capacity*150)+(N_bookings*150)+(price-self.min_budget)
                    wheres = np.append(wheres,arguement)
                    each_step=np.append(each_step,0)
                continue
            revenue_manager.add_bookings(bookings_df)

            #Save every state-action visited over the episode
            for i in range(len(bookings_df)):
                N_bookings += 1
                price = bookings_df[i][6][0]
                rev +=price

                argument = (((self.gamma*8)-extra_days)*self.capacity*150)+(N_bookings*150)+(price-self.min_budget)
                
                wheres = np.append(wheres,argument)
                each_step=np.append(each_step,price)



        #Update Q-values
        for i in range(len(wheres)):
            self.state[wheres[i]][1] +=1
            self.value_DOW[day][wheres[i]][1] +=1
            self.value_month[month][wheres[i]][1] +=1

            rev_discount = 0
            gamma=1
            Nstep=0
            lambd=0.5
            for j in range(len(each_step[i:-1])):
                #rev_discount += (gamma**j)*each_step[i+j]
                k=np.arange(0,j+1)
                
                l = lambd**(k-1)*gamma**k * each_step[i:i+j+1]
                Q = (self.state[wheres[j+1]][0]+self.value_DOW[day][wheres[j+1]][0]+self.value_month[month][wheres[j+1]][0])/3.
                Nstep += np.sum(l) + Q
               
            alpha = 1./self.state[wheres[i]][1]
            alpha_DOW = 1./self.value_DOW[day][wheres[i]][1]
            alpha_month = 1./self.value_month[month][wheres[i]][1]
            #print(alpha,Nstep,self.state[wheres[i]][0],(alpha) * ((1-lambd)*Nstep - self.state[wheres[i]][0]))
            self.state[wheres[i]][0] += (alpha) * ((1-lambd)*Nstep - self.state[wheres[i]][0])
            self.value_DOW[day][wheres[i]][0] += (alpha_DOW) * ((1-lambd)*Nstep - self.value_DOW[day][wheres[i]][0])
            self.value_month[month][wheres[i]][0] += (alpha_month) * ((1-lambd)*Nstep - self.value_month[month][wheres[i]][0])
        return revenue_manager

    #@profile
    def generate_bookings(self, time,rates_table_np,person_np,extra_days):
        bookings_df = None

        #person_np_book = person_np[person_np[:,4]==time]
        person_np_book = person_np[person_np[:,3]==extra_days]
        #person_np_book_1 = person_np[person_np[:,6]==extra_days]
        #print(len(person_np[:,4]),len(person_np[:,6]))
        #print(len(person_np_book_1),len(person_np_book))
        for i in range(len(person_np_book)):
            a = Person(budget=person_np_book[i,0],possible_start_dates=person_np_book[i,1],     possible_end_dates=person_np_book[i,2],possible_booking_lead_times=person_np_book[i,3],type_of_person='transient')
            temp_bookings_df = a.generate_bookings(time,rates_table_np)
            if temp_bookings_df is None:
                continue
            if bookings_df is None:
                bookings_df = temp_bookings_df
            else:
                bookings_df = np.append(bookings_df,temp_bookings_df,axis=0)
        return bookings_df



        """for person in persons:

            temp_bookings_df = person.generate_bookings(time, rates_table_np)

            if temp_bookings_df is None:
                continue
            if bookings_df is None:
                bookings_df = temp_bookings_df
            else:
                bookings_df = np.append(bookings_df,temp_bookings_df,axis=0)
        return bookings_df"""

    #@profile
    def _create_persons(self, reserved_night_date):
        
        #where = self._first_start_dates.index(reserved_night_date)
        day = reserved_night_date.weekday()
        month = reserved_night_date.month-1
        total_potential_customers = self.average_number_of_potential_customers *self.month_prob[month]*self.dow_prob[day]
        self._budgets = self._initialize_budgets(int(total_potential_customers))
        budgets = self._budgets#[np.array(self._first_start_dates) == reserved_night_date]
        
        
        gamma_weights = [1.2,0.8,1.2,1.1,0.6,1.1,1.2,0.7,0.6,1.2,1.1,1.1]
        gamma_weights_dow = [1.15,1.15,1.1,1.1,0.8,0.8,0.8]
        gamma = self.gamma*gamma_weights[reserved_night_date.month-1]*gamma_weights_dow[reserved_night_date.weekday()]

       	lead_days_dist = lead_days_distribution(gamma, int(gamma * 8))


        for i in range(len(budgets)):
            # create individual person

            choice = np.random.rand()

            x = np.arange(0,gamma*8)
            lead_days_dist = gamma**2/(gamma**2 + x**2)
            lead_days_to_use = np.random.choice(x,
                                                p=lead_days_dist / np.sum(lead_days_dist))
            length_of_stay = 1 # for now we just do an LOS of 1
            budget = budgets[i]
            
            start_dates = [reserved_night_date]


            possible_start_dates=np.array(start_dates)
            possible_end_dates=np.array([start_date+dt.timedelta(days=length_of_stay) for start_date in start_dates])
            possible_booking_lead_times=np.array([lead_days_to_use])

            lead_dates = np.array([start_date - dt.timedelta(days=lead_days_to_use) for start_date in start_dates])
            
            days_since =(int(gamma*8)-(lead_days_to_use))
            

            new_person = np.array([budget,possible_start_dates,possible_end_dates,possible_booking_lead_times,lead_dates,'transient',days_since])
            
            if i ==0:
                person_np = new_person
            else:
                person_np=np.vstack([person_np,new_person])
            
            
            #pers_writer.writerow([lead_days_to_use,budget])
        return person_np

 

    @staticmethod
    #@profile
    def _calculate_day(year, month,cumulative_sum_weekday):
        
        weekday = len(cumulative_sum_weekday[cumulative_sum_weekday < np.random.uniform()])
        possible_weekdays = [day for day in range(1, monthrange(year, month)[1] + 1) if dt.datetime(year, month, day).weekday() == weekday]
        cumulative_sum = np.arange(0, 1.001, 1/len(possible_weekdays))[1:]
        return possible_weekdays[len(cumulative_sum[cumulative_sum < np.random.uniform()])]

    @staticmethod
    #@profile
    def _calculate_month(uniform_distribution_value,cumulative_sum):
        #cumulative_sum = np.cumsum([month_probability(i) for i in range(1, 13)])
        return len(cumulative_sum[cumulative_sum < uniform_distribution_value]) + 1

    def _calculate_year(self, uniform_distribution_value):
        years = range(self.start_reserved_night_date.year, self.end_reserved_night_date.year+1)
        if len(years) == 1:
            return self.start_reserved_night_date.year
        cumulative_sum = np.cumsum(np.arange(0, 1, 1/len(years)))
        return years[len(cumulative_sum[cumulative_sum < uniform_distribution_value])]

    def _fill_generation_statistics(self):
        self._number_of_potential_customers_per_month = [0] * 12
        for date in self._first_start_dates:
            month_index = date.month - 1
            self._number_of_potential_customers_per_month[month_index] += 1

