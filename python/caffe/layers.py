from .proto import caffe_pb2
from google import protobuf

class Function:
    def __init__(self, type_name, inputs, params):
        self.type_name = type_name
        self.inputs = inputs
        self.params = params

    def to_proto(self):
        return self._to_proto(caffe_pb2.NetParameter(), {})

    def _gen_name(self, names):
        names[self.type_name] = names.get(self.type_name, 0) + 1
        name = self.type_name + str(names[self.type_name])
        return name

    def _to_proto(self, net, names):
        bottom_names = []
        for inp in self.inputs:
            inp._to_proto(net, names)
            bottom_names.append(net.layers[-1].top[0])
        layer = net.layers.add()
        layer.type = getattr(caffe_pb2.LayerParameter, self.type_name.upper())
        layer.bottom.extend(bottom_names)
        layer.top.append(self._gen_name(names))
        for k, v in self.params.iteritems():
            setattr(getattr(layer, self.type_name + '_param'), k, v)
        return net

class Layers:
    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            return Function(name, args, kwargs)
        return layer_fn

class Parameters:
    def __getattr__(self, name):
       class Param:
            def __getattr__(self, param_name):
                return getattr(getattr(caffe_pb2, name.capitalize() + 'Parameter'), param_name)
       return Param()

layers = Layers()
params = Parameters()
