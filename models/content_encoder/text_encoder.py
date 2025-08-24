import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    DEVICE_TYPE = "npu"
except ModuleNotFoundError:
    DEVICE_TYPE = "cuda"


class TransformersTextEncoderBase(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(
        self,
        text: list[str],
    ):
        device = self.model.device
        batch = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        output: BaseModelOutput = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = output.last_hidden_state
        mask = (attention_mask == 1).to(device)

        return {"output": output, "mask": mask}


class T5TextEncoder(TransformersTextEncoderBase):
    def __init__(self, model_name: str = "/hpc_stor03/sjtu_home/yaoyun.zhang/model_ckpts/google-flan-t5-large"):
        nn.Module.__init__(self)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()

    def forward(
        self,
        text: list[str],
    ):
        with torch.no_grad(), torch.amp.autocast(
            device_type=DEVICE_TYPE, enabled=False
        ):
            return super().forward(text)


if __name__ == '__main__':
    text_encoder = T5TextEncoder()
    text = ["dog barking and cat moving"]
    text_encoder.eval()
    with torch.no_grad():
        output = text_encoder(text)
    print(output["output"].shape)
    #print(output)