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
