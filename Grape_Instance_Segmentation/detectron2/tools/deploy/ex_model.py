import torch

# identity_module = torch.nn.Identity()

# torch.save(identity_module, "identity_module.pth")
# model = torch.load("identity_module.pth")
# trace_model = torch.jit.trace(model, torch.randn(3, 3))
# torch.jit.save(trace_model, "trace_model.ts")
# tracee = torch.jit.load('model.ts')
# ts_model = tracee.to("cuda").eval()
# print(tracee)

print(torch.distributed.is_available())