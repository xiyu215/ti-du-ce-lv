# ti-du-ce-lv

用 [dezero](https://github.com/oreilly-japan/deep-learning-from-scratch-3)（《深度学习入门 3：从零开始构建框架》）手写的 **Actor-Critic**，在 `CartPole-v0` 上训练。

## 内容

单文件 `actor_critic.py`：

- `PolicyNet`：两层全连接（128 → action_size），输出经 softmax 的动作概率
- `ValueNet`：两层全连接（128 → 1），输出状态价值
- `Agent`：策略网络与价值网络各配一个 Adam 优化器，`update()` 中按 TD 误差做单步更新
  - 价值网络损失：`mean_squared_error(v, reward + γ·v(next_state)·(1 - done))`
  - 策略网络损失：`-log π(a|s) · (target - v)`，其中 `target`、`delta` 已 `unchain()` 截断梯度

采用**每步更新**（one-step Actor-Critic），而非回合结束后批量更新。

训练结束后脚本会再开一局**演示**：以 `render_mode='human'` 新建环境，按贪心动作（`argmax`）走完整回合，窗口以 50 FPS 播放，结束后打印该局回报并关闭窗口。演示用贪心动作而非采样，是为了让行为稳定、便于观察学到的策略。

## 运行

```bash
uv sync
uv run actor_critic.py
```

默认训练 3000 回合，每 100 回合打印一次 `total_reward`；训练完毕后弹出 CartPole 窗口演示一局。

## 环境

| 依赖 | 版本 |
| --- | --- |
| Python | >= 3.13 |
| dezero | >= 0.0.13 |
| gym | >= 0.26.2 |
| numpy | >= 2.5.5 |
| matplotlib | >= 3.11.2 |
| pygame | >= 2.6.1 |

## 兼容性说明

文件开头两行是针对 NumPy 2.x 的补丁：`dezero` 与 `gym 0.26.2` 仍在引用 `np.bool8` / `np.int` 等已在 NumPy 2.0 移除的别名，直接运行会抛 `AttributeError`。

```python
np.bool8 = np.bool_
np.int = int
```

## 备注

`reward_history` 已收集各回合回报，但脚本未绘制曲线；如需学习曲线可在训练循环后补一段 matplotlib 绘图。
