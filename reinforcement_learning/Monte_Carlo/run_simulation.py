import datetime as dt
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hotel import Hotel
from market import Market
from revenue_managers import RLRevenueManager,LeadRevenueManager, GammaRevenueManager

# Monte-Carlo Reinforcement Learning (Simpler and quicker than N-step)

@profile
def run_simulation(run,value,value_DOW,value_month):
    ###parameters
    set_price=71
    gamma = 20
    average_number_of_potential_customers = 100
    min_budget=30
    capacity = 76
    max_lead=200

    ###Weights
    month_prob = [0.75,0.8,0.95,0.9,1.,0.95,0.95,1.,1.,1.,1.1,1.2]
    dow_prob = [1,1,1,1,1,1,0.7]
    gamma_weights = [1.1,0.8,1.2,1.1,0.6,1.1,1.2,0.9,0.7,1.0,0.9,0.9]
    gamma_weights_dow = [1.05,1.05,1.05,1.05,0.9,0.95,0.8]

    ###Start/End date
    start_reserved_night_date = dt.datetime(2018, 1, 1)
    end_reserved_night_date = dt.datetime(2018, 12, 30)


    ###Define Hotel. (Doesn't do much now, but in future will have multiple merchants/room types)
    hotel = Hotel(merchant_id=1, capacities={10: capacity}, available_rooms=[10])
    
    #Initial prices for each DOW
    prices = {
        10: {0: set_price, 1: set_price, 2: set_price, 3: set_price, 4: set_price, 5: set_price, 6: set_price},
    }
    ###Open and write first line of output files
    """with open("transformed_data.csv", "w") as data:
        writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['hotelroom_id','reserved_night_date','booking_datetime','type','price'])
    person_data=open("person_data.csv", "w")
    pers_writer = csv.writer(person_data, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
    pers_writer.writerow(['booking_datetime','price'])"""
    
    ###Call revenue manager (currently very simple)
    lead_rm = RLRevenueManager(prices=prices, hotel=hotel, start_reserved_night_date=start_reserved_night_date,
                           end_reserved_night_date=end_reserved_night_date)


    start_date = start_reserved_night_date
    end_date = end_reserved_night_date

   
   
    ### initialise number of customers, budget and interested date
    market = Market(
        start_reserved_night_date=start_reserved_night_date,
        end_reserved_night_date=end_reserved_night_date,
        average_number_of_potential_customers=average_number_of_potential_customers,
        min_budget=min_budget,gamma=gamma,gamma_weights_months=gamma_weights,
gamma_weights_dow=gamma_weights_dow,month_prob=month_prob,dow_prob=dow_prob,capacity=capacity,
state=value,value_DOW=value_DOW,value_month=value_month
    )
    #market.initialize_market()


    ### iterate over days
    maximum_wtps = {}
    for delta in range((end_date-start_date).days):
        reserved_night_date = start_date + dt.timedelta(days=delta)
        
        #print("Processing", reserved_night_date)
        
        person_np = market._create_persons(reserved_night_date)#,pers_writer)
        #person_np[:,6]==delta
        #print(person_np[:,6],delta)
        lead_rm = market.generate_bookings_for_reserved_night_date(reserved_night_date, lead_rm, person_np,run)



    print('Total Visits',np.sum(market.state[:,1]))
    

    #More output writing
    """bookings= np.array([[0,dt.datetime(2018,1,1),0]]*len(lead_rm._current_bookings))
    bookings[:,0] = lead_rm._current_bookings[:,2]
    bookings[:,1] = lead_rm._current_bookings[:,0]
    bookings[:,2] = lead_rm._current_bookings[:,1]
    df= pd.DataFrame(data=bookings)
    df.columns=['hotelroom_id','reserved_night_date','units']
    df.to_csv('transformed_room_nights.csv',index=False)
    person_data.close()"""
    return lead_rm._revenue,market.state,market.value_DOW,market.value_month

if __name__ == '__main__':
    rm_revs = []
    gamma_rm_revs = []

 
    price_range =150
    max_lead=200
    capacity=76
    min_budget=30
    
    #Create empty Q-value tables
    value = np.array([np.array([0,0])]*max_lead*capacity*price_range)
    value_M = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_T = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_W = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Th = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_F = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_S = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Su = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Jan = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Feb = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Mar = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Apr = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_May = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Jun = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Jul = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Aug = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Sep = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Oct = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Nov = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))
    value_Dec = np.array([np.array([0,0])]*(max_lead)*(capacity)*(price_range))

    value_DOW = [value_M,value_T,value_W,value_Th,value_F,value_S,value_Su]
    value_month =[value_Jan,value_Feb,value_Mar,value_Apr,value_May,value_Jun,value_Jul,
value_Aug,value_Sep,value_Oct,value_Nov,value_Dec]

    #Load Q-value table for index
    """index=14000
    value=np.loadtxt('value_%i.txt'%index)#%(j*100 +100))
    value_M = np.loadtxt('valueDOW0_%i.txt'%index)
    value_T = np.loadtxt('valueDOW1_%i.txt'%index)
    value_W = np.loadtxt('valueDOW2_%i.txt'%index)
    value_Th = np.loadtxt('valueDOW3_%i.txt'%index)
    value_F = np.loadtxt('valueDOW4_%i.txt'%index)
    value_S = np.loadtxt('valueDOW5_%i.txt'%index)
    value_Su = np.loadtxt('valueDOW6_%i.txt'%index)
    value_Jan = np.loadtxt('valuemonth0_%i.txt'%index)
    value_Feb = np.loadtxt('valuemonth1_%i.txt'%index)
    value_Mar = np.loadtxt('valuemonth2_%i.txt'%index)
    value_Apr = np.loadtxt('valuemonth3_%i.txt'%index)
    value_May = np.loadtxt('valuemonth4_%i.txt'%index)
    value_Jun =np.loadtxt('valuemonth5_%i.txt'%index)
    value_Jul = np.loadtxt('valuemonth6_%i.txt'%index)
    value_Aug =np.loadtxt('valuemonth7_%i.txt'%index)
    value_Sep = np.loadtxt('valuemonth8_%i.txt'%index)
    value_Oct = np.loadtxt('valuemonth9_%i.txt'%index)
    value_Nov = np.loadtxt('valuemonth10_%i.txt'%index)
    value_Dec = np.loadtxt('valuemonth11_%i.txt'%index)
    value_DOW = [value_M,value_T,value_W,value_Th,value_F,value_S,value_Su]
    value_month =[value_Jan,value_Feb,value_Mar,value_Apr,value_May,value_Jun,value_Jul,value_Aug,
value_Sep,value_Oct,value_Nov,value_Dec]"""

    
    #value=np.loadtxt('value.txt')
    for i in range(1000):
        print("Run: ", i)

        gamma_rm_rev,value,value_DOW,value_month = run_simulation(i,value,value_DOW,value_month)
        #rm_revs.append(rm_rev)
        print('Total Revenue:',gamma_rm_rev)
        gamma_rm_revs.append(gamma_rm_rev)
        #a=np.array([1,2,4,6,8,10,12,14,16,18,20])*100
        a=np.array([1,2,3,4,5,6,7,8,9,10])*1000

        #Save Q-value tables
        """if i+1 in a:
            #np.savetxt('value_%i_Nstep.txt'%(i+1),value,fmt='%i')
            ##np.savetxt('revenue_9000_exp.txt',gamma_rm_revs)
            index = (i+1)+10000
            np.savetxt('value_%i.txt'%index,value,fmt='%i')
            np.savetxt('valueDOW0_%i.txt'%index,value_DOW[0],fmt='%i')
            np.savetxt('valueDOW1_%i.txt'%index,value_DOW[1],fmt='%i')
            np.savetxt('valueDOW2_%i.txt'%index,value_DOW[2],fmt='%i')
            np.savetxt('valueDOW3_%i.txt'%index,value_DOW[3],fmt='%i')
            np.savetxt('valueDOW4_%i.txt'%index,value_DOW[4],fmt='%i')
            np.savetxt('valueDOW5_%i.txt'%index,value_DOW[5],fmt='%i')
            np.savetxt('valueDOW6_%i.txt'%index,value_DOW[6],fmt='%i')
            np.savetxt('valuemonth0_%i.txt'%index,value_month[0],fmt='%i')
            np.savetxt('valuemonth1_%i.txt'%index,value_month[1],fmt='%i')
            np.savetxt('valuemonth2_%i.txt'%index,value_month[2],fmt='%i')
            np.savetxt('valuemonth3_%i.txt'%index,value_month[3],fmt='%i')
            np.savetxt('valuemonth4_%i.txt'%index,value_month[4],fmt='%i')
            np.savetxt('valuemonth5_%i.txt'%index,value_month[5],fmt='%i')
            np.savetxt('valuemonth6_%i.txt'%index,value_month[6],fmt='%i')
            np.savetxt('valuemonth7_%i.txt'%index,value_month[7],fmt='%i')
            np.savetxt('valuemonth8_%i.txt'%index,value_month[8],fmt='%i')
            np.savetxt('valuemonth9_%i.txt'%index,value_month[9],fmt='%i')
            np.savetxt('valuemonth10_%i.txt'%index,value_month[10],fmt='%i')
            np.savetxt('valuemonth11_%i.txt'%index,value_month[11],fmt='%i')"""

        #plt.plot(np.arange(0,len(gamma_rm_revs)),gamma_rm_revs)
        #plt.savefig('revenue_14000_lft.png')
        #plt.close()
     
    np.savetxt('revenue_1000.txt',gamma_rm_revs)
