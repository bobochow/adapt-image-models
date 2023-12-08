import torch

# 假设你有一个大小为(T, H, W)的张量 tensor
# 你想要取出 [ :t, :x, :y ] 部分的子张量

# 生成一个示例张量
B, T, H, W = 2, 2, 4, 4
x = torch.randn(B, T, H, W)


print("原始张量: ")
print(x)

x0 = x[:, :, 0::2, 0::2,]  # B D H/2 W/2 C
x1 = x[:, :, 1::2, 0::2,]  # B D H/2 W/2 C
x2 = x[:, :, 0::2, 1::2,]  # B D H/2 W/2 C
x3 = x[:, :, 1::2, 1::2,]  # B D H/2 W/2 C
x_cat = torch.cat([x0, x1, x2, x3], 0)  # 4B D H/2 W/2 C

# 将 x_cat 沿着第 0 维度分割成 4 份
x0, x1, x2, x3 = torch.split(x_cat, B, dim=0)

x_new=torch.zeros(B,T,H,W)

# 恢复 x 的形状
x_new[:, :, 0::2, 0::2,]=x0
x_new[:, :, 1::2, 0::2,]=x1
x_new[:, :, 0::2, 1::2,]=x2
x_new[:, :, 1::2, 1::2,]=x3

print("恢复后的张量: ")
print(x_new)
# 现在 x_restored 的形状应该和原始 x 一样


# 使用切片操作获取子张量和剩余的张量
# sub_tensor = tensor[:, :-2, :-2]

# remaining_tensor_a = tensor[:, -2:, -2:]
# remaining_tensor_b = tensor[:, :-2, -2:]
# remaining_tensor_c = tensor[:, -2:, :-2]

# merging_tensor_b= torch.cat((sub_tensor,remaining_tensor_b),dim=2)

# merging_tensor_ac= torch.cat((remaining_tensor_c,remaining_tensor_a),dim=2)

# merging_tensor= torch.cat((merging_tensor_b,merging_tensor_ac),dim=1)

# # 打印结果
# print("子张量:")
# print(sub_tensor)
# print("\n剩余的张量:")
# print(remaining_tensor_a)
# print("\n剩余的张量:")
# print(remaining_tensor_b)
# print("\n剩余的张量:")
# print(remaining_tensor_c)

# print("\n合并的张量b:")
# print(merging_tensor_b)

# print("\n合并的张量c,a:")
# print(merging_tensor_ac)

# print("\n合并的张量:")
# print(merging_tensor)