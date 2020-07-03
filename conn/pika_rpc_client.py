import pika
import json


class RPCClient(object):
    def __init__(self):
        self.conn = None
        self.channel = None

    def connect(self, host_name="localhost"):
        if self.channel is not None:
            return False, "A channel already exists"
        self.conn = pika.BlockingConnection(pika.ConnectionParameters(host_name))
        self.channel = self.conn.channel()

    def create_queue(self, queue_name):
        self.channel.queue_declare(queue=queue_name)

    def send_msg(self, queue_name, json_msg):
        self.channel.basic_publish(exchange="",
                                   routing_key=queue_name,
                                   body=json.dumps(json_msg))

    # def on_response(self, ch, method, props, body):
    #     if self.corr_id == props.correlation_id:
    #         self.response = body
    #
    # def call(self, queue_name, msg):
    #     self.response = None
    #     self.corr_id = str(uuid.uuid4())
    #     self.ch.basic_publish(exchange='',
    #                           routing_key=queue_name,
    #                           properties=pika.BasicProperties(reply_to=self.response_queue,
    #                                                           correlation_id=self.corr_id),
    #                           body=str(msg))
    #
    #     while self.response is None:
    #         self.conn.process_data_events()
    #     return self.response


if __name__ == '__main__':
    # rpc_client = RPCClient()
    # for i in range(100):
    #     data = {"graph":"hello", "no":12}
    #     json_str = json.dumps(data)
    #     print('Calling building_detector with data{}'.format(json_str))
    #     result = rpc_client.call('building_detector', json_str)
    #     result_data = json.loads(result)
    #     print('Result {}'.format(result_data))
    pass
