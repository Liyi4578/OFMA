# OFMA
Object-level Feature Memory and Aggregation for Live-stream Video Object Detection



![图片1](https://github.com/Liyi4578/OFMA/assets/57708904/3ed36705-7673-4039-82eb-0d1b279096ee)



训练后会在 output_files 产生模型文件，在 detrac_res 文件夹下产生评估数据（DETRAC格式的），因为训练方式的影响，会有重复结果且格式略有问题，在 evaluate.py 脚本中会进行进一步 NMS，并调整格式，然后使用官方的评估工具进行评估。详情见脚本（temp_dir 即对应训练产生的原始预测结果）。当然，这在实际使用中是不需要的，只是因为 dataloader 的原因才会这样。
