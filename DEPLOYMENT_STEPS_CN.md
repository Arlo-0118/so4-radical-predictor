# Streamlit Community Cloud 更新步骤

## 重要原则

不要删除当前应用，也不要重新 Create app。只更新原 GitHub 仓库 `chenchao-0118/so4-radical-predictor` 的 `main` 分支，现有 URL 才会保持不变：

`https://so4-radical-predictor-mu7aqdxctdvmz9s84flem2.streamlit.app`

## 需要上传的内容

将本文件夹中的以下内容覆盖到 GitHub 仓库根目录：

- `app.py`
- `predictor_core.py`
- `requirements.txt`
- `runtime.txt`
- `README.md`
- `artifacts/big_qsar_so4_v2_mixed/` 下的全部文件

`test_platform.py` 供本地审计使用，可以一并上传，但不会影响应用运行。

## 推荐方法：GitHub Desktop

1. 在 GitHub Desktop 登录拥有该仓库权限的 GitHub 账号。
2. 选择 `File > Clone repository`，克隆 `chenchao-0118/so4-radical-predictor`。
3. 打开克隆后的本地仓库文件夹。
4. 将本部署包上述文件复制进去，并确认选择“替换目标中的文件”。
5. 回到 GitHub Desktop，逐项确认更改只涉及平台文件和模型资产。
6. Summary 填写 `fix: align deployed predictor with revised risk policy`。
7. 点击 `Commit to main`，再点击 `Push origin`。

## Streamlit Cloud 重新部署

1. 登录 `https://share.streamlit.io/`。
2. 打开现有 `so4-radical-predictor` 应用。
3. 等待 GitHub 更新触发自动部署；如果没有触发，进入应用管理菜单并选择 Reboot。
4. 确认主文件仍为仓库根目录的 `app.py`，现有应用继续使用 Python 3.11；不要为了改 Python 版本删除应用。
5. 查看日志，必须出现依赖安装完成且没有 traceback。

## 上线验收

1. 输入 `CCO`，不填条件：应显示条件未评估，并明确使用缺失编码。
2. 输入 `CCO`、pH `7`、温度 `25`：应显示条件处于观察范围内，并自动计算 pH/T(K)。
3. 输入 `CCO`、pH `12`、温度 `50`：应出现条件越界和高风险警告。
4. 输入 `CCO`、pH `22`：平台必须拒绝输入，不得生成预测结果。
5. 检查结构 AD 同时显示正式阈值 `0.70` 与严重警戒线 `0.40`。
6. 检查预测结果可下载，压缩包中应含 JSON 报告和近邻 CSV。
7. 用手机流量或无痕窗口访问现有 URL，确认无需登录即可打开。
