# debug_model_return.py
import torch

from diffunet.diffunet_model import DiffUNet

model = DiffUNet(4, 4)
model.train()
model.to("cuda:0")

# Fake input
fake_image = torch.randn(1, 4, 128, 128, 128).cuda()
fake_label = torch.randint(0, 4, (1, 128, 128, 128)).cuda()

# Test training mode
result_train = model(fake_image, fake_label)
print(f"Training mode return type: {type(result_train)}")
if isinstance(result_train, tuple):
    print(f"Training mode returns {len(result_train)} values")
    for i, r in enumerate(result_train):
        if isinstance(r, torch.Tensor):
            print(f"  Value {i}: shape={r.shape}, dtype={r.dtype}")
        else:
            print(f"  Value {i}: type={type(r)}, value={r}")
elif isinstance(result_train, torch.Tensor):
    print(f"Training mode returns 1 tensor: shape={result_train.shape}, dtype={result_train.dtype}")
else:
    print(f"Training mode returns: {result_train}")

# Test inference mode
model.eval()
with torch.no_grad():
    result_infer = model(fake_image, ddim=True)
    print(f"\nInference mode return type: {type(result_infer)}")
    if isinstance(result_infer, torch.Tensor):
        print(f"Inference mode returns 1 tensor: shape={result_infer.shape}, dtype={result_infer.dtype}")
    elif isinstance(result_infer, tuple):
        print(f"Inference mode returns {len(result_infer)} values")
        for i, r in enumerate(result_infer):
            if isinstance(r, torch.Tensor):
                print(f"  Value {i}: shape={r.shape}")