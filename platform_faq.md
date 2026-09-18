
### 测试时间太长
1. 很多代码中会需要下载预训练模型，考虑将模型本地下载好，传到平台后指定模型路径，提交测试时一起提交上去，避免让平台去下载国外源的模型
2. 考虑将模型加载放在 predict 函数外面，避免每次 predict 都加载模型
3. 若模型太大导致 OOMkilled，会有概率没有结果返回
4. 优化 predict 函数中的模型和代码，包括模型大小、结构、数据处理代码

### 重新进入项目卡 99%
**原因**：新安装的包导致环境起不来  
**解决方法**：

+ **方法一（预防）**：不少同学在安装依赖于 `Pydantic` 的包时，同时安装了 `dataclasses`，会导致此问题。可在终端运行以下命令删除 `dataclasses`：

```shell
rm -rf .localenv/lib/python3.9/site-packages/dataclasses*
```

+ **方法二（问题已发生）**：点击右上角重启按钮，并勾选“重置 Python 环境”

### 新装包时部分依赖安装失败
**原因**：环境由持久化和非持久化两个部分组成，新装包会在持久化环境中，以确保这些包在项目重启后不会丢失。但若依赖中有非持久化的包，这些依赖则会安装失败。  
**解决方法**：

+ 如果没有造成包之间的版本兼容性问题，新装包可以正常运行；
+ 如果造成了版本兼容性问题，有两个解决方法：
    1. 使用 `/home/jovyan/.virtualenvs/basenv/bin/pip` 升级非持久化环境中安装失败/冲突的包
    2. 使用 `pip` 降级持久化环境中的包以解决兼容问题

### 离线（CPU/GPU）任务运行结果文件无法获取
**解决方法**：将结果文件指定到 `results` 目录

### 离线（CPU/GPU）任务无法读取到 Notebook 里可以读取的文件
**原因：**Notebook 的工作目录是 `./`，离线任务的工作目录是`/home/joyvan/work`

**解决方法**：使用绝对路径，或使用 `__file__`动态获取脚本所在目录，再 join 接相对路径

### Notebook 或离线任务 OOM（被 Kill）
**原因及解决方法**：

+ **数据处理任务**：尝试分块处理数据
+ **训练任务**：
    - 尝试减少 batch size，如果影响模型性能，可考虑使用梯度累积
    - 通过优化模型架构、模型量化、剪枝、混合精度等方法解决 OOM 问题
    - 如果以上方法均无效，可通过随机采样减小数据集体积

### 自测样例代码
```python

def test_case1():
    data_map = {'cloudy':['/home/jovyan/work/datasets/67fc7ccbb88b01da6626732d-momodel/train/cloudy/cloudy_00001.jpg'],
               'rainy':['/home/jovyan/work/datasets/67fc7ccbb88b01da6626732d-momodel/train/rainy/rainy_00001.jpg']}

    y_true = []
    y_pred = []

    for truth, file_paths in data_map.items():
        for file_p in file_paths:
            img = cv2.imread(file_p, 1)
            output = self.main.predict(img)
            y_true.append(truth)
            y_pred.append(output)

    # 计算得分
    score = f1_score(y_true, y_pred, average='macro')
    return score
```

### 提交测试时若缺少包/库，可在提交文件`main.py`最顶上自行安装
比如，添加如下代码

```python
import os
os.system('pip install my_package==0.0.1')

# 剩余代码
```
