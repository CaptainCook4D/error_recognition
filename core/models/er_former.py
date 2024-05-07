from core.models.blocks import EncoderLayer


class ErFormer:

    def __init__(self):
        input_dimension = 1024
        step_encoder = EncoderLayer(d_model=1024, dim_feedforward=2048, nhead=8, batch_first=True)
        pass

    def forward(self):
        pass
