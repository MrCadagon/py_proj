# if elif else == != > < >= <=
cars=['audi','bmw','toyota']
for car in cars:
    if car == 'bmw':
        print("this is bmw")
    elif car == 'audi':
        print("this is audi")
    else :
        print("others")

if cars[0] != 'bmw':
    print("cars[0] is not bmw")

test=100
if test !=101:
    print("111")
if test >101:
    print("111")
if test <101:
    print("111")
if test >=101:
    print("111")

# and or in notin
age_0=10
age_1=20
if age_0>10 and age_1<20:
    print(1)
if age_0>10 or age_1<20:
    print(1)

if 'bmw' in cars:
    print("I have bmw")
if 'bmw' not in cars:
    print("I have bmw")

#判断列表是否为空
test_array=[]
if test_array:
    print("this array is not empty")
else:
    print("this array is empty")



