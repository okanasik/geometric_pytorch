import pika


def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


def connect():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    return channel


def define_queues(ch, queue_names):
    for queue_name in queue_names:
        ch.queue_declare(queue=queue_name)


def listen_queues(ch, queue_names, callbacks):
    for i, queue_name in enumerate(queue_names):
        ch.basic_consume(queue=queue_name, on_message_callback=callbacks[i])
    ch.start_consuming()


def cb_fib(ch, method, props, body):
    n = int(body)
    print(' Calculate fib({})'.format(n))
    result = fib(n)
    print(' Result fib({})={}'.format(n, result))
    ch.basic_publish(exchange='', routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(result))
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == '__main__':
    ch = connect()
    queue_names = ['fib']
    define_queues(ch, queue_names)
    callbacks = [cb_fib]
    define_queues(ch, queue_names)
    listen_queues(ch, queue_names, callbacks)
