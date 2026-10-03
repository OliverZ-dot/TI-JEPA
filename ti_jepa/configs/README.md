这一阶段没有用 yaml config，训练超参直接走 `train.py` 的 CLI 参数（见根目录
`README.md` 的复现命令）。迁回官方 `le-wm` 规模、需要管理更多超参组合时，
再把 `train_inertia.yaml` / `train_pusht.yaml` 之类的配置放在这里。
