# Reinforcement_learning
**程序安装**

你不需要在你的计算机上安装 OpenAI Gym，你可以在课堂里完成所有的编程实现过程。你可以通过查看该 [GitHub 代码库](https://github.com/openai/gym.git)详细了解 OpenAI Gym。

建议你花时间查看 [leaderboard](https://github.com/openai/gym/wiki/Leaderboard)，其中包含每个任务的最佳解决方案。

请参阅此[博客帖子](https://blog.openai.com/openai-gym-beta/)，详细了解如何使用 OpenAI Gym 加速强化学习研究。

安装说明 （可选）
如果你想在你的计算机上安装 OpenAI Gym，建议你完成以下简单安装过程：

``` python
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```
安装 OpenAI Gym 后，请获取经典控制任务（例如“CartPole-v0”)的代码：

``` python
pip install -e '.[classic_control]'
```
最后，通过运行在 examples 目录中提供的[简单的随机智能体](https://github.com/openai/gym/blob/master/examples/agents/random_agent.py)检查你的安装情况。

``` python
cd examples/agents
python random_agent.py
```
（这些说明摘自该 [GitHub 代码库](https://github.com/openai/gym) 中的自述文件。）

在这门课程中，我们将摘录这本[关于强化学习的经典教科书](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/suttonbookdraft2018jan1.pdf)中的章节。

注意，所有建议的阅读资料都是可选阅读内容！

**参考书**

请参阅此 [GitHub 代码库](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)以查看该教科书中的大多数图表的 Python 实现。

在学习下节课之前，建议你阅读[该教科书](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/suttonbookdraft2018jan1.pdf)的第一章节（尤其是第 1.1-1.4 部分），以便了解关于强化学习领域的背景知识。

**参考指南**

建议你下载[此表格](https://github.com/udacity/rl-cheatsheet/blob/master/cheatsheet.pdf)，其中包含我们将在这门课程中使用的所有记法和算法。请仅将此表格作为你的笔记补充内容！:)

你还可以在[该教科书](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/suttonbookdraft2018jan1.pdf)第一章节之前的页面中找到另一个实用记法指南。

**玩象棋**

假设你是一个智能体，你的目标是玩象棋。在每个时间步，你都从游戏中的一组可能的走法中选择任何一个动作。你的对手是环境的一部分；你以自己的走法做出回应，你在下个时间步收到的状态是当你选择下个走法时棋盘的布局。奖励仅在游戏结束时获得，假设如果你获胜了，奖励为 1，失败了，奖励为 -1。

这是一个阶段性任务，当游戏结束时，一个阶段结束。原理是通过玩该游戏很多次，或通过与该环境互动很多个阶段，你越来越善于玩象棋。

需要注意的是，这个问题非常难，因为只有游戏结束时才会获得反馈。如果你失败了（并在阶段结束时获得奖励 -1），不清楚你到底何时出错了：或许你玩的很差，每步都出错了，或者你大部分时间都玩的很好，只是在结束时犯了一个小小的错误。

在这种情形下，奖励提供的信息非常少，我们称这种任务存在稀疏奖励问题。这是一个专门的研究领域，如果感兴趣的话，建议你详细了解一下。

**有限MDP**
请使用[此链接](https://github.com/openai/gym/wiki/Table-of-environments)获取 OpenAI Gym 中的可用环境。

### 总结
**设置，重新经历**

- 强化学习 (RL) 框架包含学习与其**环境**互动的**智能体**。
- 在每个时间步，智能体都收到环境的**状态**（环境向智能体呈现一种情况），智能体必须选择相应的响应**动作**。一个时间步后，智能体获得一个**奖励**（环境表示智能体是否对该状态做出了正确的响应）和新的**状态**。
- 所有智能体的目标都是最大化预期**累积奖励**，或在所有时间步获得的预期奖励之和。

**阶段性任务与连续性任务**
- **任务**是一种强化学习问题。
- **连续性任务**是一直持续下去、没有结束点的任务。
- **阶段性任务**是起始点和结束点明确的任务。
  - 在这种情况下，我们将一个完整的互动系列（从开始到结束）称为一个**阶段**。
  - 每当智能体抵达**最终状态**，阶段性任务都会结束

**奖励假设**
- **奖励假设**：所有目标都可以构建为最大化（预期）累积奖励。
**目标和奖励**
-（请参阅第 1 部分和第 2 部分，以查看在现实问题中如何指定奖励信号的示例。）

**累积奖励**
- **在时间步** t 的回报是<a href="https://www.codecogs.com/eqnedit.php?latex=G_t&space;:=&space;R_{t&plus;1}&space;&plus;&space;R_{t&plus;2}&space;&plus;&space;R_{t&plus;3}&space;&plus;&space;\ldots" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G_t&space;:=&space;R_{t&plus;1}&space;&plus;&space;R_{t&plus;2}&space;&plus;&space;R_{t&plus;3}&space;&plus;&space;\ldots" title="G_t := R_{t+1} + R_{t+2} + R_{t+3} + \ldots" /></a>
- 智能体选择动作的目标是最大化预期（折扣）回报。（注意：折扣将在下部分讲解。）

**折扣回报**
- 在时间步t的折扣回报是 <a href="https://www.codecogs.com/eqnedit.php?latex=G_t&space;:=&space;R_{t&plus;1}&space;&plus;&space;\gamma&space;R_{t&plus;2}&space;&plus;&space;\gamma^2&space;R_{t&plus;3}&space;&plus;&space;\ldots" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G_t&space;:=&space;R_{t&plus;1}&space;&plus;&space;\gamma&space;R_{t&plus;2}&space;&plus;&space;\gamma^2&space;R_{t&plus;3}&space;&plus;&space;\ldots" title="G_t := R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots" /></a>
- 折扣回报 γ 是你设置的值，以便进一步优化智能体的目标。
  - 它必须指定 0≤γ≤1。
  - 如果 γ=0，智能体只关心最即时的奖励。
  - 如果 γ=1，回报没有折扣。
  - γ 的值越大，智能体越关心遥远的未来。γ 的值越小，折扣程度越大，在最极端的情况下，智能体只关心最即时的奖励。

**MDPs和一步动态特性**
- **状态空间**<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{S}" title="\mathcal{S}" /></a>是所有（非终止）状态的集合。
  - 在阶段性任务中，我们使用<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{S}^&plus;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{S}^&plus;" title="\mathcal{S}^+" /></a> 表示所有状态集合，包括终止状态。
- **动作空间** <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" title="\mathcal{A}" /></a>是潜在动作的集合。 (此外， <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" title="\mathcal{A}" /></a>(s)是指在状s∈<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{S}" title="\mathcal{S}" /></a>的潜在动作集合。)
- (请参阅第 2 部分，了解如何在回收机器人示例中指定奖励信号。)
- 环境的**一步动态特性**会判断环境在每个时间步如何决定状态和奖励。可以通过指定每个潜在 s', r, s, and  a 的 <a href="https://www.codecogs.com/eqnedit.php?latex=p(s',r|s,a)&space;\doteq&space;\mathbb{P}(S_{t&plus;1}=s',&space;R_{t&plus;1}=r|S_{t}&space;=&space;s,&space;A_{t}=a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(s',r|s,a)&space;\doteq&space;\mathbb{P}(S_{t&plus;1}=s',&space;R_{t&plus;1}=r|S_{t}&space;=&space;s,&space;A_{t}=a)" title="p(s',r|s,a) \doteq \mathbb{P}(S_{t+1}=s', R_{t+1}=r|S_{t} = s, A_{t}=a)" /></a> 定义动态特性。
- 一个**（有限）马尔可夫决策过程 (MDP)** 由以下各项定义：
  - 一组（有限的）状态 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{S}" title="\mathcal{S}" /></a>（对于阶段性任务，则是 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{S}^&plus;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{S}^&plus;" title="\mathcal{S}^+" /></a>）
  - 一组（有限的）动作 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" title="\mathcal{A}" /></a>
  - 一组奖励<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{R}" title="\mathcal{R}" /></a>
  - 环境的一步动态特性
  - 折扣率  [0,1]γ∈[0,1]
