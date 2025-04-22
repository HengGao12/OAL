'''
    Use this file to calculate the FLOPs/GFLOPs and parameters of a model.  :)
'''
import torch
import torchvision.models as models
from thop import profile
from timm.models.vision_transformer import VisionTransformer
from torchinfo import summary
from openood.networks.resnet18_224x224 import ResNet18_224x224
 
# load the model

# load vit-b-16
# model = VisionTransformer(num_classes=100)
# model.load_state_dict(
#     torch.load('./results/cifar100_vit_pretrained_finetune_trainer_e50_lr0.0001_default_cifar100_finetune_final_2/s0/best.ckpt')
# )

# load resnet18
model = ResNet18_224x224(num_classes=100)
model.load_state_dict(
    torch.load('./results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt')
)

model.eval() 


# 使用thop计算FLOPs和参数数量
inputs = torch.randn(1, 3, 224, 224)  # 创建一个随机的输入张量
flops, params = profile(model, inputs=(inputs,))
print(f'FLOPs: {flops}')

# 计算GFLOPs
gflops = flops / 1e9
print(f'GFLOPs: {gflops}')

print(f'Params: {params}')
 
# 使用torchinfo打印模型摘要（包括FLOPs的近似值）
summary(model, input_data=inputs)