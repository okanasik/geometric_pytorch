import pika
import json


class PikaRPCServer():
    def __init__(self):
        self.channel = None
        # self.model = None
        self.connection = None
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.callback_method = None

    def connect(self, host_name="localhost"):
        if self.channel is not None:
            return False, "A channel already exists"
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host_name))
        self.channel = self.connection.channel()
        return True, "Success"

    def listen_queue(self, queue_name, callback_method):
        self.callback_method = callback_method
        self.channel.queue_declare(queue=queue_name)
        self.channel.basic_consume(queue=queue_name,
                                   on_message_callback=lambda ch, m, p, b: self.callback(ch, m, p, b))

    def callback(self, ch, method, props, body):
        json_object = json.loads(body)
        cb_result = self.callback_method(json_object)
        cb_result = json.dumps(cb_result)

        ch.basic_publish(exchange='', routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id=props.correlation_id),
                         body=cb_result)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_listening(self):
        self.channel.start_consuming()

    def create_queue(self, queue_name):
        self.channel.queue_declare(queue=queue_name)

    def send_msg(self, queue_name, json_msg):
        self.channel.basic_publish(exchange="",
                                   routing_key=queue_name,
                                   body=json.dumps(json_msg))

    def stop_listening(self):
        # self.channel.stop_consuming()
        # self.channel.close()
        self.connection.close()


if __name__ == '__main__':
    model_server = PikaRPCServer()
    model_server.connect()
    # model_server.serve_model("building_detector", "../rescue/models/topk_test_gat.pt")
    print("Start listening for building_detector queue")
