import datetime as dt
import csv
import attr
import numpy as np
import pandas as pd


@attr.s
class BaseRevenueManager(object):
    hotel = attr.ib()
    prices = attr.ib()
    start_reserved_night_date = attr.ib()
    end_reserved_night_date = attr.ib()

    _current_bookings = attr.ib(default=None)
    _revenue = attr.ib(default=0)
    _sales_per_month = attr.ib(default=None)
    _revenue_per_month = attr.ib(default=None)
    _unavailable_rooms = attr.ib(default=attr.Factory(dict))

    def get_price(self, time, reserved_night_date, hotelroom_id):
        pass

    #@profile
    def get_rates_table_for_reserved_night_date(self, time, reserved_night_date,state,N_bookings,lead,run,value_DOW,value_month):
        prices = []
        hotelroom_ids = []
        for hotelroom_id in self.hotel.available_rooms:
            price = self.get_price(time, reserved_night_date, hotelroom_id,state,N_bookings,lead,run,value_DOW,value_month)
            #print("rm.price",price)
            if price is None:
                continue
            prices.append(price)
            hotelroom_ids.append(hotelroom_id)
        """result = pd.DataFrame(
                {
                    'merchant_id': [self.hotel.merchant_id] * len(prices),
                    'hotelroom_id': hotelroom_ids,
                    'reserved_night_date': [reserved_night_date] * len(prices),
                    'price': prices,
                }
            )"""
        #print("merchant_id",[self.hotel.merchant_id]*len(prices),[hotelroom_ids],[prices])
        #print("date",reserved_night_date)
        result_np = np.array([[self.hotel.merchant_id]*len(prices),hotelroom_ids,[reserved_night_date]*len(prices),prices]).T
        #print("result_np",result_np)
        #print("result",result)
        return result_np

    #@profile
    def add_bookings(self, bookings_df):
        if self._sales_per_month is None:
            self._sales_per_month = [0] * 12
        if self._revenue_per_month is None:
            self._revenue_per_month = [0] * 12
        #print(bookings_df)
	#bookings_df = [lead_date, hotelroom_id,merchant_id,reserved_night_date,environment,segment,price]
        for (row) in bookings_df:
            segment = row[5]
            if segment is not 'transient':
                continue
            reserved_night_date = row[3]
            hotelroom_id = row[1]
            if hotelroom_id not in self.hotel.available_rooms:
                continue
            if self._current_bookings is None:
                self._initialize_current_bookings()


            element = np.logical_and(self._current_bookings[:,0]==reserved_night_date, self._current_bookings[:,2]==hotelroom_id)
            
            """index = self._current_bookings.index[np.logical_and(
                self._current_bookings[:,2] == hotelroom_id,
                self._current_bookings[:,0] == reserved_night_date
            )].tolist()[0]"""
           
            self._current_bookings[:,1][element] += 1
            self._revenue += row[6]
            sale_month_index = reserved_night_date.month - 1
            self._sales_per_month[sale_month_index] += 1
            self._revenue_per_month[sale_month_index] += row[6]
            if self._is_room_full(hotelroom_id, self._current_bookings[:,1][element]):
                self._update_unavailable_hotelrooms(hotelroom_id, reserved_night_date)

            #More output 
            """with open("transformed_data.csv", "a") as data:
                writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([row[1],row[3],row[0],'booking',row[6][0]])"""


    def _update_price_change(self, reserved_night_date, hotelroom_id, price_change):
        """index = self._current_bookings.index[np.logical_and(
            self._current_bookings['hotelroom_id'] == hotelroom_id,
            self._current_bookings['reserved_night_date'] == reserved_night_date
        )].tolist()[0]"""

        element = np.logical_and(self._current_bookings[:,0]==reserved_night_date, self._current_bookings[:,2]==hotelroom_id)
        print("CB_check",self._current_bookings)
        self._current_bookings[:,3][element] = price_change
        #self._current_bookings.at[index, 'price_change'] = price_change

    def _initialize_current_bookings(self):
        reserved_night_dates = []
        hotelroom_ids = []
        for hotelroom_id in self.hotel.available_rooms:
            for i in range((self.end_reserved_night_date-self.start_reserved_night_date).days + 1):
                reserved_night_dates.append(self.start_reserved_night_date+dt.timedelta(days=i))
                hotelroom_ids.append(hotelroom_id)
        """self._current_bookings = pd.DataFrame({
            'reserved_night_date': reserved_night_dates,
            'current_bookings': [0] * len(reserved_night_dates),
            'hotelroom_id': hotelroom_ids,
            'price_change': [0] * len(reserved_night_dates)
        })"""
        self._current_bookings = np.array([reserved_night_dates,[0]*len(reserved_night_dates),hotelroom_ids,[0]*len(reserved_night_dates)]).T

    def _is_room_unavailable(self, reserved_night_date, hotelroom_id):
        if hotelroom_id not in self._unavailable_rooms.keys():
            return False
        unavailable_room_nights = self._unavailable_rooms[hotelroom_id]
        if reserved_night_date not in unavailable_room_nights:
            return False
        return True

    def _is_room_full(self, hotelroom_id, number_of_bookings):
        if self.hotel.capacities[hotelroom_id] <= number_of_bookings:
            return True
        else:
            return False

    def _update_unavailable_hotelrooms(self, hotelroom_id, reserved_night_date):
        if hotelroom_id not in self._unavailable_rooms.keys():
            self._unavailable_rooms[hotelroom_id] = [reserved_night_date]
        else:
            self._unavailable_rooms[hotelroom_id].append(reserved_night_date)


@attr.s
class DOWRevenueManager(BaseRevenueManager):

    #@profile
    def get_price(self, time, reserved_night_date, hotelroom_id):
        if hotelroom_id not in self.hotel.available_rooms:
            raise Exception
        if self._is_room_unavailable(reserved_night_date, hotelroom_id):
            return None
        day_of_week = reserved_night_date.weekday()
        return self.prices[hotelroom_id][day_of_week]


@attr.s
class GammaRevenueManager(BaseRevenueManager):

    gamma = attr.ib(default=10)
    drop_increase_fraction = attr.ib(default=0.1)
    target_occupancy = attr.ib(default=1)

    def get_price(self, time, reserved_night_date, hotelroom_id):
        if hotelroom_id not in self.hotel.available_rooms:
            raise Exception
        if self._is_room_unavailable(reserved_night_date, hotelroom_id):
            return None
        day_of_week = reserved_night_date.weekday()
        lead_time = (reserved_night_date-time) / dt.timedelta(days=1)

        if lead_time > self.gamma:

            return self.prices[hotelroom_id][day_of_week]
        if self._current_bookings is None:
            self._initialize_current_bookings()
        current_bookings_for_room = self._current_bookings[self._current_bookings[:,2] == hotelroom_id]
        current_bookings = current_bookings_for_room[:,1][current_bookings_for_room[:,0]==reserved_night_date]

        price_change = current_bookings_for_room[:,3][current_bookings_for_room[:,0]==reserved_night_date][0]
    

        if price_change == 0:

            price_change = 1 if current_bookings / self.hotel.capacities[hotelroom_id] > 0.5 * self.target_occupancy else -1
            self._update_price_change(reserved_night_date, hotelroom_id, price_change)
        #print("grm.prices",self.prices[hotelroom_id][day_of_week],price_change)
        return self.prices[hotelroom_id][day_of_week] * (1 + self.drop_increase_fraction * price_change)

@attr.s
class LeadRevenueManager(BaseRevenueManager):
    def get_price(self,time,reserved_night_date,hotelroom_id):
        if hotelroom_id not in self.hotel.available_rooms:
            return Exception
        if self._is_room_unavailable(reserved_night_date,hotelroom_id):
            return None
        month = reserved_night_date.month-1
        day_of_week = reserved_night_date.weekday()
        lead_time = (reserved_night_date-time)/dt.timedelta(days=1)

        #change_time =[65,23,7,5,3,0
        price_change_month =[0.8,0.9,0.9,1.05,1.05,1.1,1.1,1.1,1.1,1.1,1.,0.95]
        #price_change =[1,28/30.,27./30,26./30,23./30,21./30]
        price_change_day=[0.9,0.95,1,1,1.05,1.05,0.95]
        #change_time =[60,57,53,35,23,5,0]
        #price_change =[1,28/30.,25./30,22./30,24./30,26./30,20./30]
        #price_change_high =[1,28/30.,25./30,22./30,25./30,28./30,30./30]
        
        rand = np.random.normal(1.05,0.4)
        if rand<0.5:
            rand=1
        return rand * self.prices[hotelroom_id][day_of_week]*price_change_month[month]*price_change_day[day_of_week]
        """for i in range(len(change_time)):
            if lead_time > change_time[i]:
                if day_of_week==4 or day_of_week==5:
                    return self.prices[hotelroom_id][day_of_week] * price_change[i]
                else:
                    return self.prices[hotelroom_id][day_of_week] * price_change[i]"""
@attr.s
class RLRevenueManager(BaseRevenueManager):
    def get_price(self,time,reserved_night_date,hotelroom_id,value,N_bookings,lead,run,value_DOW,value_month):
        if hotelroom_id not in self.hotel.available_rooms:
            return Exception
        if self._is_room_unavailable(reserved_night_date,hotelroom_id):
            return None
        month = reserved_night_date.month-1
        day_of_week = reserved_night_date.weekday()
        lead_time = (reserved_night_date-time)/dt.timedelta(days=1)
 
        capacity = 76
        min_budget=30
        

        explore = np.random.rand()
        price=0
        #Epsilon greedy policy improvement
        if explore > 0.001*run:
             return np.random.choice(np.arange(30,175))
        else:
            arguement = (lead*capacity*150)+(N_bookings*150)+(0)
            arguement2 = (lead*capacity*150)+(N_bookings*150)+(150)
            #a=value[:,0][arguement:arguement2].argmax()
            a=(value[:,0][arguement:arguement2]+value_DOW[day_of_week][:,0][arguement:arguement2]+value_month[month][:,0][arguement:arguement2]/3).argmax()
            return a+30



