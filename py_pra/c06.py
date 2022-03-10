# 字典
alien={
    'color':'green',
    'points':5
}
print(alien['color'])
alien['x_position']=10
alien['color']='red'
del alien['points']

print(alien)

# 遍历字典
for key,value in alien.items():
    print(key)
    print(value)
print("\n")

# for key in alien.keys():
for key in alien:
    print(key)
print("\n")

for value in alien.values():
    print(value)



