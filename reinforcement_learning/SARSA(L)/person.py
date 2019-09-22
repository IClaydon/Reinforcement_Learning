import datetime as dt
import csv
import attr
import numpy as np
import pandas as pd

AVAILABLE_PERSON_TYPES = ['transient', 'group', 'corporate']
AVAILABLE_CHANNEL_TYPES = ['Booking.com', 'Expedia', 'Website', 'Phone', 'Tripadvisor', 'Agoda', 'Walk-In']

BOOKING_COLUMN_NAMES = [
    'booking_datetime', 'hotelroom_id', 'merchant_id', 'reserved_night_date', 'channel_id', 'segment_id', 'price']

@attr.s
class Person(object):
    budget = attr.ib()
    possible_start_dates = attr.ib()
    possible_end_dates = attr.ib()
    possible_booking_lead_times = attr.ib()
    type_of_person = attr.ib()
    has_booked = attr.ib(default=False)

    #@profile
    def generate_bookings(self, time,rates_table_np):
        if self.has_booked:
            return None
        lead_times =[0]*len(self.possible_start_dates)
        
        for i in range(len(self.possible_start_dates)):
            
            lead_times[i] = (self.possible_start_dates[i] - time).days #/ dt.timedelta(days=1)
        
        best_bookable_budgets_np = None

        for i, lead_time in enumerate(lead_times):
            if lead_time in self.possible_booking_lead_times:
                

                budgets_np = rates_table_np[np.logical_and(
                        rates_table_np[:,2] >= self.possible_start_dates[i],
                        rates_table_np[:,2] < self.possible_end_dates[i]
                    )
                ]
                

                if len(budgets_np) == 0:
                    continue


                argmin = budgets_np[:,3].argmin()
                min_budget_np=budgets_np[argmin,3]
                hotelroom_id_np = budgets_np[argmin,1]
                merchant_id_np = budgets_np[argmin,0]
                
                new_df_np = np.append(budgets_np[budgets_np[:,3].argmin()],[self.possible_start_dates[i],self.possible_end_dates[i]])
                
                if best_bookable_budgets_np is None:
                     best_bookable_budgets_np = np.array([new_df_np])
                     
                else:
                    best_bookable_budgets_np=best_bookable_budgets_np.append(new_df_np)


        best_budget = self._select_best_budget(best_bookable_budgets_np)
        
        if best_budget is None or len(best_budget) == 0:
            return None

        self.has_booked = True

        return self._generate_bookings_from_selected_merchant(
            best_budget,
            time,
            rates_table_np
        )

    @staticmethod
    def _filter_rooms_full_for_some_dates(rates_table, start_date, end_date):
        length_of_stay = (end_date-start_date).days
        available_rates = rates_table.groupby(['hotelroom_id']).count().reset_index()
        #print('here',available_rates)
        available_hotelrooms = available_rates[available_rates['price'] == length_of_stay]['hotelroom_id'].values
        #print('here',available_hotelrooms)
        return rates_table[rates_table['hotelroom_id'].isin(available_hotelrooms)]

    #@profile
    def _select_best_budget(self, best_bookable_budgets_np):
        # pretty simple for now
        if best_bookable_budgets_np is None or len(best_bookable_budgets_np) == 0:
            return None
        if best_bookable_budgets_np[:,3].min() > self.budget:
            return None
        return best_bookable_budgets_np[best_bookable_budgets_np[:,3].argmin()]

    #@profile
    def _generate_bookings_from_selected_merchant(self, best_budget, booking_datetime, rates_table_np):
        channel = AVAILABLE_CHANNEL_TYPES[int(np.floor(np.random.uniform(len(AVAILABLE_CHANNEL_TYPES))))]
        prices = []
        reserved_night_dates = []
        start_date = best_budget[4]
        end_date = best_budget[5]
        for i in range((end_date-start_date).days):
            reserved_night_date = start_date + dt.timedelta(days=i)
            """price = rates_table[
                np.logical_and(rates_table['reserved_night_date'] == reserved_night_date,
                               rates_table['hotelroom_id'] == hotelroom_id
                               )
            ]['price'].values[0]"""
            price = rates_table_np[:,3][rates_table_np[:,2]==reserved_night_date and rates_table_np[:,1]==best_budget[1]]
            #print("here",price[0])
            prices.append(price)
            reserved_night_dates.append(reserved_night_date)
        """result = pd.DataFrame(
            {
                BOOKING_COLUMN_NAMES[0]: [booking_datetime] * len(prices),
                BOOKING_COLUMN_NAMES[1]: [best_budget[1]] * len(prices),
                BOOKING_COLUMN_NAMES[2]: [best_budget[0]] * len(prices),
                BOOKING_COLUMN_NAMES[3]: reserved_night_dates,
                BOOKING_COLUMN_NAMES[4]: [channel] * len(prices),
                BOOKING_COLUMN_NAMES[5]: [self.type_of_person] * len(prices),
                BOOKING_COLUMN_NAMES[6]: prices
            }
        )"""
        result_np = np.array([[booking_datetime]*len(prices),[best_budget[1]] * len(prices),[best_budget[0]] * len(prices),reserved_night_dates,[channel] * len(prices),[self.type_of_person] * len(prices),prices]).T
        #print("PRINT",prices[0][0])
        
            
        return result_np
