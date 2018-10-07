## OpenAI Gym：FrozenLakeEnv
在 FrozenLake 环境中，智能体探索的是 4x4 网格世界。你可以打开相应的[GitHub](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py) 文件并转到 FrozenLakeEnv 类的注释部分，详细了解该环境。澄清下，我们也摘录了该环境的说明，如下所示：

```
    冬天来了。你和朋友在公园里投掷飞盘，你一不小心将飞盘远远地扔到了湖中心。水几乎都结冰了，但是有几个融化的洞口。如果你踏入了其中一个洞口，将掉进冰冷的水中。现在全球的飞盘都处于紧缺状态，因此你必须在湖面上拿到飞碟。但是冰面很滑，因此你并不能一直朝着期望的方向行走。冰面用以下网格形式表示
        SFFF
        FHFH
        FFFH
        HFFG
    S：起始点，安全
    F：冰冻湖面，安全
    H：洞口，跌入湖中
    G：目标，拿到飞盘
    当你抵达目的地或跌入湖中，这一阶段结束。
    当你抵达目的地时，获得奖励 1，否则获得奖励 0。

```

## 动态规划设置
OpenAI Gym 中的环境以强化学习设置为依据。因此，OpenAI Gym 不允许轻松地访问马尔可夫决策流程 (MDP) 的底层一步动态特性。

为了使用动态规划设置的 FrozenLake 环境，我们首先需要下载包含 FrozenLakeEnv 类的[文件](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)。然后，我们添加一行代码，以便将 MDP 的一步动态规划与智能体分享。
```
# obtain one-step dynamics for dynamic programming setting
self.P = P
```
然后将新的 FrozenLakeEnv 类保存到 Python 文件 frozenlake.py 中，我们将使用该文件（而不是原始 OpenAI Gym 文件）创建该环境的实例。

