# #函数
# def greet_user():
#     print("hello")
#
# def greet_user2(username):
#     print("hello "+username)
#
# greet_user()
# greet_user2("tom")
#
# # 位置实参 关键字实参
# def describe_pets(name,pet):
#     print(name+" have a pet(name:"+pet+")")
# describe_pets('tom','fee')
# describe_pets(pet='fee2',name='tom2')
#
# # 形参的默认值
# def describe_pets_2(name,pet='doggy'):
#     print(name + " have a pet(name:" + pet + ")")
# describe_pets_2('tom')
#
# #返回值
# def get_formatted_name(first_name,last_name):
#     return first_name+' '+last_name
# print(get_formatted_name('tom','riddle'))

#work1
def print1(unpint,printed):
    print('test-------')
    while unpint:
        printed.append(unpint.pop())

def show_models(models):
    for model in models:
        print(model)

unprint=['a','b','c']
printed=[]
print1(unprint,printed)
print('the unprint is:')
show_models(unprint)
print('the printed is:')
show_models(printed)

#work 1.2使用 实参副本
unprint2=['a','b','c']
printed2=[]
print1(unprint2[:],printed2)
print('the unprint2 is:')
show_models(unprint2)
print('the printed2 is:')
show_models(printed2)

#任意数量的形参[列表 字典]
def make_pizza(size,*pizzas):
    print("this size is:"+str(size))
    for pizza in pizzas:
        print("this pizza is:"+pizza)
make_pizza(10,'apple','cheess','meat')

def make_prfile(fn,ln,**info):
    print("the name is"+fn+' '+ln)
    for key,value in info.items():
        print("the "+key+' is '+value)

print()
make_prfile('tom','riddle',field='physics',home='china')









