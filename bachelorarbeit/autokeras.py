import autokeras as ak
from autokeras.utils import utils, layer_utils
from tensorflow.keras import layers
from tensorflow.python.util import nest


class PaddedConvBlock(ak.ConvBlock):
    """
    Der PaddedConvBlock ist eine eigene Implementierung des ConvBlocks aus der AutoKeras Bibliothek.
    Er limitiert die Anzahl der Filter pro Block und benutzt padding="same" statt padding="valid" um die Größe
    des Spielfelds über mehrere Schichten des Netzwerks beizubehalten. Mit padding="valid" würde das Spielfeld bereits
    nach zwei Schichten auf 1x2 Felder schrumpfen.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        num_blocks = self.num_blocks or hp.Choice("num_blocks", [1, 2, 3], default=2)
        num_layers = self.num_layers or hp.Choice("num_layers", [1, 2], default=2)
        separable = self.separable
        if separable is None:
            separable = hp.Boolean("separable", default=False)

        if separable:
            conv = layer_utils.get_sep_conv(input_node.shape)
        else:
            conv = layer_utils.get_conv(input_node.shape)

        max_pooling = self.max_pooling
        if max_pooling is None:
            max_pooling = hp.Boolean("max_pooling", default=True)
        pool = layer_utils.get_max_pooling(input_node.shape)

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        for i in range(num_blocks):
            kernel_size = self.kernel_size or hp.Choice(
                "kernel_size_{i}".format(i=i)
                , [3, 5], default=3
            )
            for j in range(num_layers):
                output_node = conv(
                    hp.Choice(
                        "filters_{i}_{j}".format(i=i, j=j),
                        [16, 32, 64, 128, 256],
                        default=32,
                    ),
                    kernel_size=kernel_size,
                    padding=hp.Choice(
                        "padding_{i}_{j}".format(i=i, j=j),
                        ["same", self._get_padding(kernel_size, output_node)],
                        default="same"
                    ),
                    activation="relu",
                )(output_node)
            if max_pooling:
                output_node = pool(
                    kernel_size - 1,
                    padding=self._get_padding(kernel_size - 1, output_node),
                )(output_node)
            if dropout > 0:
                output_node = layers.Dropout(dropout)(output_node)
        return output_node
