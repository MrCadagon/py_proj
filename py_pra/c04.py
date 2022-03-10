nums=['num1','num2','num3','num4']
for num in nums:
    print(num)
for i in range(1,5):
    print(i)
for i in [1,2,7]:
    print(i)

list_nums=list(range(1,10))
list_nums_even=list(range(1,10,2))
print(list_nums_even)

# 乘方**
squares=[]
for i in range(1,11):
    squares.append(i**2)
print(squares)

#min max sum
print(min(squares))
print(max(squares))
print(sum(squares))

# 列表解析
set1=[i**3 for i in range(1,11)]
print(set1)

# 切片
set_cut=['num1','num2','num3','num4','num5']
print(set_cut[0:3])
print(set_cut[3:])
print(set_cut[:3])
for cut in set_cut[:3]:
    print(cut)

# 复制构造
set_cut_2=set_cut[:]
# 指向同一变量 引用
set_cut_3=set_cut


# 元组
dimensions=(100,200)
# 不可修改
# dimensions[0]=20
print(dimensions)

# 给元祖初始化OK
dimensions=(300,400,500)













print(1)