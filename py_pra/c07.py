# # 输入输出
# message=input("Hi tom\ninput the message :")
# print(message)
# #转换为整数型
# age=input("input your age:")
# age=int(age)
# print("your age in double is "+str(age*2))
#
# # 求模
# if age%2==0:
#     print("your age is even")

#while
# count_num=1
# while count_num<5:
#     print("count is "+str(count_num))
#     count_num+=1

# # break
# while True:
#     message=input("input msg, ('quit' to quit)")
#     if message.lower()=='quit':
#         break
#     else:
#         print(message)

# #continue
# count_num=-1;
# while count_num<1000:
#     count_num+=1
#     if count_num%2==1:
#         continue
#     print(count_num)

# #while处理列表字典1 pop append
# unconfirmed_users=['A','B','C','D']
# confirmed_users=[]
# while unconfirmed_users:
#     now_user=unconfirmed_users.pop()
#     confirmed_users.append(now_user)
#     print("appeend:"+now_user)

# #while处理列表字典1 remove
# pets=['cats','dog','cats','cats','dog','cats','cats']
# while 'cats' in pets:
#     print("remove cats")
#     # 只删除第一个元素
#     pets.remove('cats')
# print(pets)

# #test 课后题1
# sandwich_orders=['fruit','pastra','sausage','pastra','cheese','pastra']
# finished_sandwiches=[]
# while sandwich_orders:
#     now_sand=sandwich_orders.pop()
#     print("this is "+now_sand+"sandwich")
#     finished_sandwiches.append(now_sand)
# print("that is all sandwiches")
# print(finished_sandwiches)
# while 'pastra' in finished_sandwiches:
#     finished_sandwiches.remove('pastra')
# print(finished_sandwiches)

#test 课后题2
name_and_visis={}
flag='conitnue'
while flag.lower()!='quit':
    name=input("input your name ")
    visit=input("input your visit place ")
    name_and_visis[name]=visit
    flag=input("input quit stop")
print(name_and_visis)