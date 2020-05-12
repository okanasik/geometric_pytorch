import pika
import uuid


class RPCClient(object):
    def __init__(self):
        self.conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.ch = self.conn.channel()

        self.response_queue = self.ch.queue_declare(queue='', exclusive=True).method.queue
        self.ch.basic_consume(queue=self.response_queue, on_message_callback=self.on_response, auto_ack=True)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, queue_name, msg):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.ch.basic_publish(exchange='',
                              routing_key=queue_name,
                              properties=pika.BasicProperties(reply_to=self.response_queue,
                                                              correlation_id=self.corr_id),
                              body=str(msg))

        while self.response is None:
            self.conn.process_data_events()
        return self.response


if __name__ == '__main__':
    rpc_client = RPCClient()
    for i in range(100):
        print('Calling fib({})'.format(i))
        result = rpc_client.call('fib', i)
        print('Result fib({})={}'.format(i, result))

