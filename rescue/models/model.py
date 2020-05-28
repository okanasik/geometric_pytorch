import torch
import sys


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.name = None
        self.version = None
        self.num_features = None
        self.num_classes = None
        self.class_name = type(self).__name__
        self.class_definition = self.get_class_definition()

    @staticmethod
    def load(file_name):
        state_dict = torch.load(file_name)
        model_name = state_dict["name"]
        # version = state_dict["version"]
        num_features = state_dict["num_features"]
        num_classes = state_dict["num_classes"]
        class_name = state_dict["class_name"]
        class_definition = state_dict["class_definition"]

        exec(class_definition, globals())
        model = eval("{}({}, {})".format(class_name, num_features, num_classes))

        del state_dict["name"]
        del state_dict["version"]
        del state_dict["num_features"]
        del state_dict["num_classes"]
        del state_dict["class_name"]
        del state_dict["class_definition"]
        model.load_state_dict(state_dict)
        return model

    def mysave(self, file_name):
        state_dict = self.state_dict()
        state_dict["name"] = self.name
        state_dict["version"] = self.version
        state_dict["num_features"] = self.num_features
        state_dict["num_classes"] = self.num_classes
        state_dict["class_name"] = self.class_name
        state_dict["class_definition"] = self.class_definition
        torch.save(state_dict, file_name)

    def get_class_definition(self):
        full_path = sys.modules[self.__module__].__file__
        with open(full_path) as fp:
            file_content = fp.read()
        return file_content







