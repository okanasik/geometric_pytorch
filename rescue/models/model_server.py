from conn.pika_rpc_server import PikaRPCServer
from rescue.models.model import Model
import rescue.dataset.data_adapter as data_adapter
import torch


class ModelServer(object):
    def __init__(self):
        self.rpc_server = PikaRPCServer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def connect(self, host_name="localhost"):
        self.rpc_server.connect(host_name)

    def serve_model(self, model_file_name, queue_name):
        self.model = Model.load(model_file_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.rpc_server.listen_queue(queue_name, self.predict)
        print("model:" + model_file_name + " start listening at " + queue_name)
        self.rpc_server.start_listening()

    def predict(self, json_object):
        data, node_indexes, node_ids = data_adapter.convert_to_graph_data(json_object["graph"],
                                                                           json_object["ambulances"],
                                                                           json_object["firebrigades"],
                                                                           json_object["polices"],
                                                                           json_object["civilians"],
                                                                           json_object["agent"])
        data_adapter.add_batch(data)
        data = data.to(self.device)
        output = self.model(data)
        pred = output.max(dim=1)[1]
        target_id = node_ids[pred]
        return target_id


if __name__ == "__main__":
    model_server = ModelServer()
    model_server.connect()
    model_server.serve_model("topk_gat_test0_rl.pt", "building_detector")


