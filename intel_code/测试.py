import numpy as np
# print(a.shape)
model_sj=[]
num_n=0
class Variable(object):
    def __init__(self,size,loc=0,std=0.1,hs=None,name=None):
        self.size=size
        self.name=name
        self.hs=hs
        self.loc=loc
        self.std=std
        self.content=np.random.normal(self.loc, self.std, self.size)
        self.shape=self.content.shape
    def text(self):
        return self.content
    def reshape(self,shape):
        self.content=self.content.reshape(shape)
        self.shape=self.content.shape


class Activations(object):
    def __init__(self,x,activation='relu',name=None):
        self.activation=activation
        self.x = x
        self.name=name
        self.content=self.__choose()
        self.save=self.__save()
    def __relu(self,x):
        a=np.copy(x)  # 用x会直接改变x本身
        a[a<0]=0
        return a
    def __softmax(self,x):
        exp=np.exp(x).T
        ex=np.exp(x)
        return (exp/np.sum(ex,axis=1)).T
    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def __choose(self):
        if self.activation=='relu':
            return self.__relu(self.x)
        elif self.activation=='tanh':
            return np.tanh(self.x)
        elif self.activation=='sigmoid':
            return self.__sigmoid(self.x)
        elif self.activation=='softmax':
            return self.__softmax(self.x)
    def __save(self):
        '''
        model_act = {'name': self.name, 'shape': self.content.shape,'data':self.content,'act':self.activation,'type':3} #求偏导，不转置
        model_sj.append(model_act)
        '''
        dx=DAct(self.content).run()
        model_1 = {'name': self.name, 'input_data': self.x, 'dx': dx,'type':1}
        model_sj.append(model_1)
class model(object):
    def __init__(self):
        pass
    def add(self,layer):
        pass
class Dense(object):
    def __init__(self,w,input_data,name=None):
        self.w=w
        # self.activation=activation
        self.input_data=input_data
        self.name=name
    def run(self):
        global num_n
        y=np.matmul(self.input_data,self.w.content)
        num_n=num_n+1
        '''
        model_x={'name':'x','shape':self.input_data.shape,'data':self.input_data,'type':2} # 不求偏导 不转置
        model_w = {'name': self.w.name, 'shape': self.w.content.shape, 'data': self.w.content,'type':1} # 求值，中间需要转置
        model_y={'name':self.name,'shape':y.shape,'data':y,'type':0} # 求偏导，转置
        model_sj.append(model_x)
        model_sj.append(model_w)
        model_sj.append(model_y)
        '''
        model_1={'name':self.name,'input_data':self.input_data,'w':self.w.content,'dx':self.w.content.T,'dw':self.input_data,'type':0}
        model_sj.append(model_1)
        return y
class DAct(object):
    def __init__(self,x,activation='relu'):
        self.x=x
        self.activation=activation
    def __drelu(self,x):
        a=np.copy(x)
        # print(a)
        a[a>0]=1
        a[a<0]=0
        return a
    def run(self):
        if self.activation=='relu':
            return self.__drelu(self.x)
class comple(object):
    def __init__(self,input_data,out_data,y_,loss,learing,epch=10):
        self.input_data=input_data
        self.out_data=out_data
        self.y_=y_  # 预测真实值
        self.loss=loss
        self.learing=learing
        self.epch=epch
    def losss(self):
        if self.loss=='mse':
            return 0.5*(self.out_data-self.y_)**2,self.learing*(self.y_-self.out_data)
    def __relu(self,x):
        a=np.copy(x)  # 用x会直接改变x本身
        a[a<0]=0
        return a
    def __drelu(self,x):
        a=np.copy(x)
        a[a>0]=1
        a[a<0]=0
        return a
    def run(self):
        global num_n
        pop=0
        while pop<self.epch:
            print('-'*30)
            pop+=1
            lit=[]
            loss, ddd = self.losss()
            print(np.sum(loss))
            for i in range(num_n):
                loss, dd = self.losss()
                j = 0
                l = len(model_sj)
                while l>0:
                    l-=1
                    if model_sj[l].get('type')==1:
                        dd=dd*model_sj[l].get('dx')
                    elif model_sj[l].get('type')==0:
                        if j==i:
                            j+=1
                            d=model_sj[l].get('dw')
                            dd=np.matmul(dd.T,d)
                            dit={'w':model_sj[l]['w'],'dw':dd.T}
                            lit.append(dit)
                            break
                        else:
                            j+=1
                            d=model_sj[l].get('dx')
                            dd = np.matmul(dd, d)
            for tmp in range(len(lit)):
                lit[tmp]['w']+=lit[tmp]['dw']
                model_sj[(len(lit)-1-tmp)*2]['w']=lit[tmp]['w']


            s=self.input_data
            p=0
            for tmp in model_sj[:-1]:
                if p%2==0:
                    y = np.matmul(s, tmp['w'])
                    tmp['x']=s
                    tmp['dx']=tmp['w'].T
                    tmp['dw']=tmp['x']
                    s=y
                else:
                    s=self.__relu(s)
                    tmp['input_data']=s
                    tmp['dx']=self.__drelu(s)
                p+=1
            # -----------------------输出层的激活函数------------------#
            s = self.__relu(s)
            self.out_data=s
            model_sj[-1]['input_data'] = s
            model_sj[-1]['dx'] = self.__drelu(s)
            # -----------------------输出层的激活函数------------------#

# a=np.array([[1,2,3,4]])
a=np.random.rand(100,2)-0.5 # (?,2)
aa=np.sum(a,axis=1).reshape(100,1) # (100,)
w1=Variable(size=(2,100),name='w1')
b=Dense(w1,a,name='b').run()
y=Activations(b,activation='relu',name='y1').content # (1,10)
w2=Variable(size=(100,1),name='w2')
c=Dense(w2,y,name='c').run()
y2=Activations(c,activation='relu',name='y2').content
z=comple(a,y2,aa,'mse',0.001,epch=1000).run()

