from torch import nn

from core.models.blocks import EncoderLayer, Encoder, MLP, fetch_input_dim


class ErFormer(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dimension = fetch_input_dim(config)

        # Initialize the transformer encoder
        step_encoder_layer = EncoderLayer(d_model=input_dimension, dim_feedforward=2048, nhead=8, batch_first=True)
        self.step_encoder = Encoder(step_encoder_layer, num_layers=1)

        # Initialize the MLP decoder
        self.decoder = MLP(input_dimension, 512, 1)

    def forward(self, input_data):
        # Encode the input
        encoded_output = self.step_encoder(input_data)

        # Decode the output
        final_output = self.decoder(encoded_output)

        return final_output

