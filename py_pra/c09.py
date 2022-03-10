# class
# class Dog():
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#     def dog_sit(self):
#         print(self.name+' has sit down!')
#
# my_dog=Dog('ben',6)
# my_dog.dog_sit()
# print()
#
# #work1
# class Restaurant():
#     def __init__(self,name,type):
#         self.name=name
#         self.type=type
#     def describe(self):
#         print('the name is '+self.name)
#         print('the cuisine type is ' + self.type)
#     def open_ressaurant(self):
#         print(self.name+' is now opening')
#
# res1=Restaurant('weea_food','seafood')
# res1.describe()
# res1.open_ressaurant()
# print()
#
# #wock 9-4 类中默认量
# class Restaurant():
#     def __init__(self,name,type):
#         self.name=name
#         self.type=type
#         self.number_served=0
#     def describe(self):
#         print('the name is '+self.name)
#         print('the cuisine type is ' + self.type)
#         print('there are '+str(self.number_served)+' in restaurant')
#     def open_ressaurant(self):
#         print(self.name+' is now opening')
#     def set_number_served(self,number_served):
#         self.number_served=number_served
#     def increse_number_served(self,incremant):
#         self.number_served+=incremant
#
# res1=Restaurant('weea_food','seafood')
# res1.describe()
# res1.open_ressaurant()
# print()

# #work2
# class User():
#     #需要在使用字典的时候新建字典
#     other_info={}
#     def __init__(self,first_n,last_n,**other_info):
#         self.first_name=first_n
#         self.last_name = last_n
#         self.age=0
#         for key,value in other_info.items():
#             self.other_info[key]=value
#
#     def describe_user(self):
#         print('the name is '+self.first_name+' '+self.last_name)
#         print('my age is '+str(self.age))
#         for key,value in self.other_info.items():
#             print('the '+key+' is '+value)
#
# class User_Boss(User):
#     def __init__(self,first_n,last_n,num_worker,**other_info):
#         super().__init__(first_n,last_n,**other_info)
#         self.num_worker=num_worker
#     def describe_num_worker(self):
#         print('the boss have '+str(self.num_worker)+' workers')
#     def describe_user(self):
#         super().describe_user()
#         self.describe_num_worker()
#
# # user=User('tom','riddle',home='china',tooth='100')
# # user.describe_user()
# boss=User_Boss('tom','riddle',200,home='china',tooth='100')
# boss.describe_user()


#work 9-6
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
res1=Restaurant('weea_food','seafood')
res1.describe()
res1.open_ressaurant()
print()

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
icecream_stand=Icecream_Stand('ice_stand','icecream','apple','banana')
icecream_stand.show()









