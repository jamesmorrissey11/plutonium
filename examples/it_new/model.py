import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


class KerasEncoder:
    def __init__(
        self,
        args,
        model_name: str,
    ):
        self.args = args
        self.model = keras.models.load_model(model_name)

    def select_intermediate_layer(self, layer_name="dense"):
        intermediate_layer_model = Model(
            inputs=self.model.input, outputs=self.model.get_layer(layer_name).output
        )
        return intermediate_layer_model

    def encode_features(self, df: pd.DataFrame):
        encoded_features = self.select_intermediate_layer().predict(K.constant(df))
        return encoded_features
