import pika
import json
from rescue.rescue_dataset import RescueDataset
from rescue.models.model import Model
import torch


class PikaRPCServer():
    def __init__(self):
        self.channel = None
        self.model = None
        self.connection = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def connect(self, host_name):
        if self.channel is not None:
            return False
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host_name))
        self.channel = self.connection.channel()

    def serve_model(self, queue_name, model_file_name):
        self.channel.queue_declare(queue=queue_name)
        self.model = Model.load(model_file_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.channel.basic_consume(queue=queue_name,
                                   on_message_callback=lambda ch, m, p, b : self.callback(ch, m, p, b))

    def callback(self, ch, method, props, body):
        json_object = json.loads(body)
        data, node_indexes, node_ids = RescueDataset.convert_to_graph_data(json_object["graph"], json_object["ambulances"],
                                            json_object["firebrigades"], json_object["polices"],
                                            json_object["civilians"], json_object["agent"])
        RescueDataset.add_batch(data)
        data = data.to(self.device)
        output = self.model(data)
        pred = output.max(dim=1)[1]
        target_id = node_ids[pred]
        ch.basic_publish(exchange='', routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id=props.correlation_id),
                         body=str(target_id))
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def listen(self):
        self.channel.start_consuming()


if __name__ == '__main__':
    model_server = PikaRPCServer()
    model_server.connect("localhost")
    model_server.serve_model("building_detector", "../rescue/models/topk_test_gat.pth")
    print("Start listening for building_detector queue")
    model_server.listen()
