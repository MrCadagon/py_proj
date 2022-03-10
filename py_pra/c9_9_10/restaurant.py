class Restaurant():
    def __init__(self,name,type):
        self.name=name
        self.type=type
        self.number_served=0
    def describe(self):
        print('the name is '+self.name)
        print('the cuisine type is ' + self.type)
        print('there are '+str(self.number_served)+' in restaurant')
    def open_ressaurant(self):
        print(self.name+' is now opening')
    def set_number_served(self,number_served):
        self.number_served=number_served
    def increse_number_served(self,incremant):
        self.number_served+=incremant