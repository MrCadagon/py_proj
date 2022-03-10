fruits=['apple','ben',"wa"]
print(fruits)
print(fruits[1])
print(fruits[-1])
print(fruits[-2])

fruits.append('append_0')
fruits.insert(0,'insert_1')

tmp=fruits.pop()
print("tmp is "+tmp)
tmp=fruits.pop(0)
print("tmp is "+tmp)

fruits.remove('apple')
print(fruits.__len__())

print(fruits)

cars=['A','B','C','D','AB']

cars.sort()
print(cars)
cars.sort(reverse=True)
print(cars)
print(sorted(cars))
cars.reverse()
print(len(cars))

print(cars)
# print(cars[5])