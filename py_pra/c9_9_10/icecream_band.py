from restaurant import Restaurant
class Icecream_Stand(Restaurant):
    flavors=[]
    def __init__(self,name,type,*flavors):
        super().__init__(name,type)
        for flavor in flavors:
            self.flavors.append(flavor)
    def show(self):
        super().describe()
        print('the flavors are:')
        print(self.flavors)