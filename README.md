# Reinforcement learning cheat sheet
在下面的链接中是强化学习基本算法的总结（来自优达学院）
https://github.com/daxingxingqi/Reinforcement_learning/blob/master/cheatsheet.pdf
# 准备
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

