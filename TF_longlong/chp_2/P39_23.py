import numpy as np
data=[]
for i in range(100):
    print(i)
    x=np.random.uniform(-10.,10.)
    eps=np.random.normal(0,0.1)
    y=1.477*x+0.089+eps
    data.append([x,y])
data=np.array(data)


def mse(b,w,points):
    error_sum=0
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error_sum+=(y-(w*x+b))**2
    return error_sum/float(len(points))

def step_gradient(b_current,w_current,points,lr):
    b_gradient=0
    w_gradient = 0
    M= float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2 / M) * (w_current * x + b_current - y)
        w_gradient += (2 / M) * (w_current * x + b_current - y) * x
    b_current-=b_gradient*lr
    w_current -= w_gradient * lr
    return [b_current, w_current]

def gradient_descent(points, starting_b, starting_w, lr, num_iter):
    b=starting_b
    w=starting_w
    for i in range(num_iter):
        b, w = step_gradient(b,w,points,lr)
        loss = mse(b, w, points)
        if i%50 == 0:
            print(f'iteration:{i} b:{b} w:{w} loss:{loss}')
    return [b,w]

def main():
    lr=0.01
    initial_b = 0
    initial_w = 0
    num_iter = 10000
    [b, w]=gradient_descent(data, initial_b, initial_w, lr, num_iter)
    loss=mse(b, w, data)
    print(f'b:{b} w:{w} loss:{loss}')

if __name__ == '__main__':
    main()













