import tensorflow as tf
import timeit

n=1000000000
with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([1,n])
    cpu_b = tf.random.normal([n,1])
    print(cpu_a.device, cpu_b.device)
def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c

with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([1, n])
    gpu_b = tf.random.normal([n, 1])
    print(gpu_a.device, gpu_b.device)
def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# cpu_time = timeit.timeit(cpu_run, number=10)
# cpu_time = timeit.timeit(cpu_run, number=10)
# print('cpu run time:', cpu_time)
gpu_time = timeit.timeit(gpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('gpu run time:', gpu_time)