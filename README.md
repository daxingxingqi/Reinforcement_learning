# Reinforcement_learning
# [Reinforcement learning cheat sheet](https://github.com/daxingxingqi/Reinforcement_learning/blob/master/cheatsheet.pdf)
## 准备
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

在学习下节课之前，建议你阅读[该教科书](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/suttonbookdraft2018jan1.pdf)，以便了解关于强化学习领域的背景知识。

**参考指南**

建议你下载[此表格](https://github.com/udacity/rl-cheatsheet/blob/master/cheatsheet.pdf)，其中包含我们将在这门课程中使用的所有记法和算法。请仅将此表格作为你的笔记补充内容！:)

你还可以在[该教科书](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/suttonbookdraft2018jan1.pdf)第一章节之前的页面中找到另一个实用记法指南。
## MDP
<div align=center><img width="650" src=resource/6.png></div>
<div align=center><img width="650" src=resource/7.png></div>
**有限MDP**
请使用[此链接](https://github.com/openai/gym/wiki/Table-of-environments)获取 OpenAI Gym 中的可用环境。

## 总结
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
- 一个 **(有限)马尔可夫决策过程(MDP)** 由以下各项定义：
  - 一组（有限的）状态 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{S}" title="\mathcal{S}" /></a>（对于阶段性任务，则是 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{S}^&plus;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{S}^&plus;" title="\mathcal{S}^+" /></a>）
  - 一组（有限的）动作 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{A}" title="\mathcal{A}" /></a>
  - 一组奖励<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{R}" title="\mathcal{R}" /></a>
  - 环境的一步动态特性
  - 折扣率  [0,1]γ∈[0,1]

## 贝尔曼方程
在这个网格世界示例中，一旦智能体选择一个动作，

- 它始终沿着所选方向移动（而一般 MDP 则不同，智能体并非始终能够完全控制下个状态将是什么）
- 可以确切地预测奖励（而一般 MDP 则不同，奖励是从概率分布中随机抽取的）。
在这个简单示例中，我们发现任何状态的值可以计算为即时奖励和下个状态（折扣）值的和。

Alexis 提到，对于一般 MDP，我们需要使用期望值，因为通常即时奖励和下个状态无法准确地预测。的确，我们在之前的课程中发现，奖励和下个状态是根据 MDP 的一步动态特性选择的。在这种情况下，奖励 r 和下个状态 s'是从（条件性）概率分布 p(s',r|s,a) 中抽取的，贝尔曼预期方程（对于 <a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi" title="v_\pi" /></a>  ）表示了任何状态 s 对于_预期即时奖励和下个状态的预期_值的值：
<a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi(s)&space;=&space;\text{}&space;\mathbb{E}_\pi[R_{t&plus;1}&space;&plus;&space;\gamma&space;v_\pi(S_{t&plus;1})|S_t&space;=s]." target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;=&space;\text{}&space;\mathbb{E}_\pi[R_{t&plus;1}&space;&plus;&space;\gamma&space;v_\pi(S_{t&plus;1})|S_t&space;=s]." title="v_\pi(s) = \text{} \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t =s]." /></a>

**计算预期值**

如果智能体的策略 π 是确定性策略，智能体在状态 s 选择动作 π(s)，贝尔曼预期方程可以重写为两个变量 (s' 和 r) 的和：
<a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi(s)&space;=&space;\sum_{s'\in\mathcal{S}^&plus;,&space;r\in\mathcal{R}}p(s',r|s,\pi(s))(r&plus;\gamma&space;v_\pi(s'))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;=&space;\sum_{s'\in\mathcal{S}^&plus;,&space;r\in\mathcal{R}}p(s',r|s,\pi(s))(r&plus;\gamma&space;v_\pi(s'))" title="v_\pi(s) = \sum_{s'\in\mathcal{S}^+, r\in\mathcal{R}}p(s',r|s,\pi(s))(r+\gamma v_\pi(s'))" /></a>

在这种情况下，我们将奖励和下个状态的折扣值之和<a href="https://www.codecogs.com/eqnedit.php?latex=(r&plus;\gamma&space;v_\pi(s'))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(r&plus;\gamma&space;v_\pi(s'))" title="(r+\gamma v_\pi(s'))" /></a>与相应的概率 p(s′,r∣s,π(s)) 相乘，并将所有概率相加得出预期值。
如果智能体的策略 π 是随机性策略，智能体在状态 s 选择动作 a 的概率是 π(a∣s)，贝尔曼预期方程可以重写为三个变量（s′ 、r 和 a）的和：

<a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi(s)&space;=&space;\sum_{s'\in\mathcal{S}^&plus;,&space;r\in\mathcal{R},a\in\mathcal{A}(s)}\pi(a|s)p(s',r|s,a)(r&plus;\gamma&space;v_\pi(s'))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;=&space;\sum_{s'\in\mathcal{S}^&plus;,&space;r\in\mathcal{R},a\in\mathcal{A}(s)}\pi(a|s)p(s',r|s,a)(r&plus;\gamma&space;v_\pi(s'))" title="v_\pi(s) = \sum_{s'\in\mathcal{S}^+, r\in\mathcal{R},a\in\mathcal{A}(s)}\pi(a|s)p(s',r|s,a)(r+\gamma v_\pi(s'))" /></a>

在这种情况下，我们将奖励和下个状态的折扣值之和 <a href="https://www.codecogs.com/eqnedit.php?latex=(r&plus;\gamma&space;v_\pi(s'))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(r&plus;\gamma&space;v_\pi(s'))" title="(r+\gamma v_\pi(s'))" /></a>与相应的概率π(a∣s)p(s′,r∣s,a) 相乘，并将所有概率相加得出预期值。

**策略**

- **确定性策略**是从 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi:&space;\mathcal{S}\to\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi:&space;\mathcal{S}\to\mathcal{A}" title="\pi: \mathcal{S}\to\mathcal{A}" /></a> 的映射。对于每个状态 <a href="https://www.codecogs.com/eqnedit.php?latex=s\in\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s\in\mathcal{S}" title="s\in\mathcal{S}" /></a>，它都生成智能体在状态 s 时将选择的动作 <a href="https://www.codecogs.com/eqnedit.php?latex=a\in\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a\in\mathcal{A}" title="a\in\mathcal{A}" /></a>

- **随机性策略**是从 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi:&space;\mathcal{S}\times\mathcal{A}\to" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi:&space;\mathcal{S}\times\mathcal{A}\to" title="\pi: \mathcal{S}\times\mathcal{A}\to" /></a> [0,1] 的映射。对于每个状态 <a href="https://www.codecogs.com/eqnedit.php?latex=s\in\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s\in\mathcal{S}" title="s\in\mathcal{S}" /></a> 和动作 <a href="https://www.codecogs.com/eqnedit.php?latex=a\in\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a\in\mathcal{A}" title="a\in\mathcal{A}" /></a>，它都生成智能体在状态s时选择动作a的概率

**状态值函数**
- 策略 π 的**状态值函数**表示为 <a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi" title="v_\pi" /></a>。对于每个状态 <a href="https://www.codecogs.com/eqnedit.php?latex=s\in\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s\in\mathcal{S}" title="s\in\mathcal{S}" /></a>，它都生成智能体从状态 s 开始，然后在所有时间步根据策略选择动作的预期回报。即 <a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi(s)&space;\doteq&space;\text{}&space;\mathbb{E}_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;\doteq&space;\text{}&space;\mathbb{E}_\pi" title="v_\pi(s) \doteq \text{} \mathbb{E}_\pi" /></a>[G_t|S_t=s]。我们将 <a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi(s)" title="v_\pi(s)" /></a> 称之为**在策略 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a> 下的状态 s 的值**。
- 记法 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{E}_\pi[\cdot]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{E}_\pi[\cdot]" title="\mathbb{E}_\pi[\cdot]" /></a>来自推荐的教科书，其中 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{E}_\pi[\cdot]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{E}_\pi[\cdot]" title="\mathbb{E}_\pi[\cdot]" /></a>定义为随机变量的预期值（假设智能体遵守策略 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a>）。

**贝尔曼方程（第 1 部分）**
<a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi" title="v_\pi" /></a> 的**贝尔曼预期方程**是：<a href="https://www.codecogs.com/eqnedit.php?latex=v_\pi(s)&space;=&space;\mathbb{E}_\pi[R_{t&plus;1}&space;&plus;&space;\gamma&space;v_\pi(S_{t&plus;1})|S_t&space;=s]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_\pi(s)&space;=&space;\mathbb{E}_\pi[R_{t&plus;1}&space;&plus;&space;\gamma&space;v_\pi(S_{t&plus;1})|S_t&space;=s]" title="v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t =s]" /></a>.

**最优性**
- 策略 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi‘" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi‘" title="\pi‘" /></a>‘定义为优于或等同于策略 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi‘" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi‘" title="\pi‘" /></a>（仅在所有 <a href="https://www.codecogs.com/eqnedit.php?latex=s\in\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s\in\mathcal{S}" title="s\in\mathcal{S}" /></a> 时 <a href="https://www.codecogs.com/eqnedit.php?latex=v_{\pi'}(s)&space;\geq&space;v_\pi(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_{\pi'}(s)&space;\geq&space;v_\pi(s)" title="v_{\pi'}(s) \geq v_\pi(s)" /></a>)。
- **最优策略** <a href="https://www.codecogs.com/eqnedit.php?latex=\pi_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_*" title="\pi_*" /></a>对于所有策略 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a> 满足 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi_*&space;\geq&space;\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_*&space;\geq&space;\pi" title="\pi_* \geq \pi" /></a>。最优策略肯定存在，但并不一定是唯一的。
- 所有最优策略都具有相同的状态值函数 <a href="https://www.codecogs.com/eqnedit.php?latex=v_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_*" title="v_*" /></a>，称为**最优状态值函数**。

**动作值函数**

- 策略 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a> 的动作值函数表示为 <a href="https://www.codecogs.com/eqnedit.php?latex=q_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_\pi" title="q_\pi" /></a>。对于每个状态<a href="https://www.codecogs.com/eqnedit.php?latex=s&space;\in\mathcal{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s&space;\in\mathcal{S}" title="s \in\mathcal{S}" /></a>和动作<a href="https://www.codecogs.com/eqnedit.php?latex=a&space;\in\mathcal{A}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a&space;\in\mathcal{A}" title="a \in\mathcal{A}" /></a>，它都生成智能体从状态s开始并采取动作a，然后在所有未来时间步遵守策略时产生的预期回报。即 <a href="https://www.codecogs.com/eqnedit.php?latex=q_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_\pi" title="q_\pi" /></a>(s,a) <a href="https://www.codecogs.com/eqnedit.php?latex=\doteq&space;\mathbb{E}_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\doteq&space;\mathbb{E}_\pi" title="\doteq \mathbb{E}_\pi" /></a>[G_t|S_t=s, A_t=a]。我们将 <a href="https://www.codecogs.com/eqnedit.php?latex=q_\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_\pi" title="q_\pi" /></a>(s,a)称之为在状态s根据策略<a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a>采取动作a的值（或者称之为**状态动作**对s,a 的值）。
所有最优策略具有相同的动作值函数 <a href="https://www.codecogs.com/eqnedit.php?latex=q_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_*" title="q_*" /></a>，称之为**最优动作值函数**。


**最优策略**

- 智能体确定最优动作值函数 <a href="https://www.codecogs.com/eqnedit.php?latex=q_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_*" title="q_*" /></a>后，它可以通过设置 <a href="https://www.codecogs.com/eqnedit.php?latex=\pi_*(s)&space;=&space;\arg\max_{a\in\mathcal{A}(s)}&space;q_*(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_*(s)&space;=&space;\arg\max_{a\in\mathcal{A}(s)}&space;q_*(s,a)" title="\pi_*(s) = \arg\max_{a\in\mathcal{A}(s)} q_*(s,a)" /></a> 快速获得最优策略 π* 。

**贝尔曼方程（第 2 部分）**

- <a href="https://www.codecogs.com/eqnedit.php?latex=q_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_*" title="q_*" /></a>的贝尔曼预期方程是：<a href="https://www.codecogs.com/eqnedit.php?latex=q_\pi(s,a)&space;=&space;\mathbb{E}_\pi[R_{t&plus;1}&space;&plus;&space;\gamma&space;q_\pi(S_{t&plus;1},A_{t&plus;1})|S_t&space;=s,&space;A_t=a]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_\pi(s,a)&space;=&space;\mathbb{E}_\pi[R_{t&plus;1}&space;&plus;&space;\gamma&space;q_\pi(S_{t&plus;1},A_{t&plus;1})|S_t&space;=s,&space;A_t=a]" title="q_\pi(s,a) = \mathbb{E}_\pi[R_{t+1} + \gamma q_\pi(S_{t+1},A_{t+1})|S_t =s, A_t=a]" /></a>.
- <a href="https://www.codecogs.com/eqnedit.php?latex=v_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_*" title="v_*" /></a>的贝尔曼最优性方程是：<a href="https://www.codecogs.com/eqnedit.php?latex=v_*(s)&space;=&space;\max_{a&space;\in&space;\mathcal{A}(s)}&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;v_*(S_{t&plus;1})&space;|&space;S_t=s]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_*(s)&space;=&space;\max_{a&space;\in&space;\mathcal{A}(s)}&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;v_*(S_{t&plus;1})&space;|&space;S_t=s]" title="v_*(s) = \max_{a \in \mathcal{A}(s)} \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s]" /></a>
- <a href="https://www.codecogs.com/eqnedit.php?latex=q_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_*" title="q_*" /></a> 的贝尔曼最优性方程是：<a href="https://www.codecogs.com/eqnedit.php?latex=q_*(s,a)&space;=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;\max_{a'\in\mathcal{A}(S_{t&plus;1})}q_*(S_{t&plus;1},a')&space;|&space;S_t=s,&space;A_t=a]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_*(s,a)&space;=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;\max_{a'\in\mathcal{A}(S_{t&plus;1})}q_*(S_{t&plus;1},a')&space;|&space;S_t=s,&space;A_t=a]" title="q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'\in\mathcal{A}(S_{t+1})}q_*(S_{t+1},a') | S_t=s, A_t=a]" /></a>
### 推导
### [动态规划推导](https://github.com/daxingxingqi/Reinforcement_learning/blob/master/dynamic/%E6%8E%A8%E5%AF%BC.ipynb)
## 蒙特卡洛方法
实例代码-https://github.com/daxingxingqi/Reinforcement_learning/tree/master/mc_predict

**MC 预测：状态值**

- 解决预测问题的算法会确定策略 π 对应的值函数 v_π（或 q_π）。
- 通过与环境互动评估策略 π 的方法分为两大类别：
  - 在线策略方法使智能体与环境互动时遵守的策略 π 与要评估（或改进）的策略相同。
  - 离线策略方法使智能体与环境互动时遵守的策略 b（其中 b≠π）与要评估（或改进）的策略不同。
  
- 状态 s∈S 在某个阶段中的每次出现称为 s 的一次经历。
- 有两种类型的蒙特卡洛 (MC) 预测方法（用于估算 v_π）：
  - 首次经历 MC 将 v_π(s) 估算为仅在 s 首次经历之后的平均回报（即忽略与后续经历相关的回报）。
  - 所有经历 MC 将 v_π(s) 估算为 s 所有经历之后的平均回报
<div align=center><img width="650" src=resource/mc-pred-state.png></div>

**MC 预测：动作值**

<div align=center><img width="650" src=resource/mc-pred-action.png></div>

**广义策略迭代**

- 旨在解决控制问题的算法会通过与环境互动确定最优策略 π∗。
- 广义策略迭代 (GPI) 是指通过交替地进行策略评估和和改进步骤搜索最优策略的广义方法。我们在这门课程中讲解的所有强化学习方法都可以归类为 GPI。

**MC 控制：增量均值**

**MC 控制：策略评估**

**MC 控制：策略改进**

**探索与利用**
<div align=center><img width="650" src=resource/mc-control-glie.png></div>

**MC 控制：常量 α**
<div align=center><img width="650" src=resource/mc-control-constant-a.png></div>
