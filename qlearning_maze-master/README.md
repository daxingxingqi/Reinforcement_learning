同时，目前强化学习在研究领域也是非常热门的方向，前有 DeepMind 的 AlphaGo 在围棋领域下赢世界顶级选手柯洁，最近 OpenAI 的 Dota 机器人已经能够操控5个AI与人类团队进行5v5对决，此外还有星际争霸2也开放了 API 用于强化学习的研究，AlphaGo Zero 更是一鸣惊人。如果你还有进一步的兴趣，可以参考如下补充资料：
- 经典读物：Reinforcement Learning: An Introduction
https://s3-us-west-1.amazonaws.com/udacity-dlnfd/suttonbookdraft2018jan1.pdf
- 伯克利 AI 课程：http://ai.berkeley.edu/lecture_videos.html
- David Silver 主讲的强化学习课程：http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
- 入门强化学的博文：https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html

## 1. 问题描述

![img.png](default.png)

在该项目中，你将使用强化学习算法，实现一个自动走迷宫机器人。

1. 如上图所示，智能机器人显示在右上角。在我们的迷宫中，有陷阱（红色炸弹）及终点（蓝色的目标点）两种情景。机器人要尽量避开陷阱、尽快到达目的地。
2. 机器人可执行的动作包括：向上走 `u`、向右走 `r`、向下走 `d`、向左走 `l`。
3. 执行不同的动作后，根据不同的情况会获得不同的奖励，具体而言，有以下几种情况。
    - 撞到墙壁：-10
    - 走到终点：50
    - 走到陷阱：-30
    - 其余情况：-0.1
4. 我们需要通过修改 `robot.py` 中的代码，来实现一个 Q Learning 机器人，实现上述的目标。

## 2. 完成项目流程

1. 配置环境，使用 `envirnment.yml` 文件配置名为 `robot-env` 的 conda 环境，具体而言，你只需转到当前的目录，在命令行/终端中运行如下代码，稍作等待即可。
```
conda env create -f environment.yml
```
安装完毕后，在命令行/终端中运行 `source activate robot-env`（Mac/Linux 系统）或 `activate robot-env`（Windows 系统）激活该环境。

2. 阅读 `robot_maze.ipynb` 中的指导完成项目，并根据指导修改对应的代码，生成、观察结果。
3. 导出代码与报告，上传文件，提交审阅并优化。

## 3. 项目review

### 问题 1：请参照如上的定义，描述出 “机器人走迷宫这个问题” 中强化学习四个组成部分对应的实际对象。
- 【状态 : 墙壁，终点，陷阱及其它情况】对状态(State)的理解有些不到位。
  - 实际上状态就是描述机器人当前情况的一个抽象概念，它与机器人能否成功学到一个策略息息相关。
  - 在我们的项目中，它是小车所处的迷宫坐标位置，例如 (0,1)、(1,1) 等。
  - 如果将其设定为“处于起点”、“处于陷阱”、“处于终点”等情况，那么思考一下：
    - 这样设置与项目中设计的区别在哪里？
     > 如果按照”墙壁，终点，陷阱及其它情况“来设置状态的话，这些信息将不能作为抽象的信息被智能体所学习。
    - 这样机器人能够学习到一个成功的策略吗？
     > 不能，因为无法作为抽象信息被计算机所学习。

### 问题 2：根据已知条件求 $q(s_{t},a)$，在如下模板代码中的空格填入对应的数字即可
- R_{t+1} 是小车在S1执行动作u之后得到的奖励，不是下S2执行动作R得到的奖励。
- 这里的下标为t+1的含义是，在t的时刻执行动作a，则会在第t+1时刻得到对应的奖励 R_{t+1}。这里是 小车在 S1 执行了动作u之后，因为没有撞到墙壁，所以获得-0.1的奖励，见报告中，Section0的第一节的第三点。
- 相应地，max_a q(a,s_{t+1}) 代表着，在 t+1 时刻，在状态 s_{t+1} 下， 对所有的动作而言，小车 q 值能够取得的最大值，也就是我们在S2下的最大的Q值。

- 实际上，此处的公式大有来头～
- 我们原本考虑的是一个动作的未来奖励，实际上是通过数学的方法，考虑在这个状态，基于执行未来n步后的奖励，通过乘上一个衰减系数 gamma，得到一个数列或者级数，然后求和得到总的未来奖励。但是实际上这样的做法既不实际（大部分情况下，你无法得到未来n步的奖励）也不实用（即便可以得到，计算的复杂度也很高）。
- 所以我们使用了如项目中所示的公式，来迭代地计算未来奖励。尽管每次的奖励会有些不准确，但是通过不断的迭代，可以比较准确地进行估计。而这就是有名的 Bellman Equation。
- 参考 David Silver 课程中的 这个ppt（http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf） 第30页。

### 问题 4：在如下的代码块中，创建你的迷宫并展示。
- 你可以尝试使用 test_world 目录下的迷宫，或者修改 Maze 类中 __generate_maze 函数中的 complexity 及 density 变量，来增加迷宫的难度～

### 问题 6：实现 Robot.py 中的8段代码，并运行如下代码检查效果。
- 请根据下方的解释，优化一下你对 Q Learning 公式的理解：
- 在如下所示 Q-learning 更新公式中，它考虑了两部分的信息：之前学习到的Q值，以及新学习到Q值。

<div align=center><img width="550" src=qlearning.svg></div>

- 在新学习到的Q值中，γ*maxQ 的一项目就考虑了所谓的「未来奖励」——这是强化学习中的一个巨大亮点。也就是说，我们在计算、衡量一个动作的时候，不仅考虑它当前一步获得的奖励 r，还要考虑它执行这个动作之后带来的累计奖励——这能够帮助我们更好地衡量一个动作的好坏。但是这时候机器人并没有真正地往前走，而是使用Qtable 中原有地 next_state 的值来估计这个未来奖励。
- 其中 γ 是折扣因子，是一个(0,1)之间的值。一般我们取0.9，能够充分地对外来奖励进行考虑。如果这个值大于1，那么实际上未来奖励会发散开来（因为这是一个不断累加、迭代的过程），导致Qtable不能发散。它能够帮助终点处的正奖励“扩散”到周围，也就是说，这样机器人更能够成功地学习到通往终点的路径。

### *问题 7：尝试利用下列代码训练机器人，并进行调参。*

【解释1】
- 首先给你补充一下对于 epsilon greedy 算法的解释：
- 对于 epsilon-greedy 算法，你可以参考论坛中的 这个帖子https://discussions.youdaxue.com/t/topic/33333：
> Q: 如何理解 greed-epsilon 方法／如何设置 epsilon／如何理解 exploration & exploitation 权衡？
A: (1) 我们的小车一开始接触到的 state 很少，并且如果小车按照已经学到的 qtable 执行，那么小车很有可能出错或者绕圈圈。同时我们希望小车一开始能随机的走一走，接触到更多的 state。(2) 基于上述原因，我们希望小车在一开始的时候不完全按照 Q learning 的结果运行，即以一定的概率 epsilon，随机选择 action，而不是根据 maxQ 来选择 action。然后随着不断的学习，那么我会降低这个随机的概率，使用一个衰减函数来降低 epsilon。(3) 这个就解决了所谓的 exploration and exploitation 的问题，在“探索”和“执行”之间寻找一个权衡。

【解释2】
- 再给你补充一下对 alpha 的解释。 alpha 是一个权衡上一次学到结果和这一次学习结果的量，如：Q = (1-alpha)*Q_old + alpha*Q_current。
- alpha 设置过低会导致机器人只在乎之前的知识，而不能积累新的 reward。一般取 0.5 来均衡以前知识及新的 reward。

【解释3】
- gamma 是考虑未来奖励的因子，是一个(0,1)之间的值。一般我们取0.9，能够充分地对外来奖励进行考虑。
- 实际上如果你将它调小了，你会发现终点处的正奖励不能够“扩散”到周围，也就是说，机器人很有可能无法学习到一个到达终点的策略。你可以自己尝试一下。

- 注意区分 `test = test**0.99` 和 `test = test*0.99`。由于 test 是一个小于 1 的数，所以作用 0.99 的指数，这个值会放大。但是 test = test*0.99 这个迭代的写法，就相当于的 `test = test0*0.99^t`，这是一个严格递减的函数。
【优化】
- 我们希望，在训练初期的时候多探索，在训练末期的时候少探索，那么我建议你选择类似 cos 这样的上凸函数。你可以查看这个链接 函数的凹凸性 （wiki 怕打不开）。这个函数的特点是，一开始导数负得小，逐渐地导数负得越来越大。你可以和指数函数对比一下。
【思考】
- 我们知道，学习率 α 的目的是为了在更新 Q 值的同时也保留过去学到的结果，那么对于不同的 state，实际上学习的进度是不一样的。那么此处对所有的 state 统一设置 α，似乎并不是最优的做法。你可以考虑对每个 state 设置不同的学习率，该 state 学习完毕后其对应的 α 衰减，而其他 state 对应的 α 不变。可以参考 周志华 的 《机器学习》（西瓜书）中相关的内容。
### 问题 8：使用 runner.plot_results() 输出训练结果，根据该结果对你的机器人进行分析。

-请进一步补充你的分析：
  - 我们希望你在这里能够更详细地说明每个参数（alpha、gamma、epsilon0 和 epsilon下降函数两者的区别和联系、训练次数）的作用是什么，它们的变化大概会怎样影响运行结果，然后有目的地对小车进行调参，比较不同参数下的训练结果，并说明你使用这个参数的原因。
  - 总结出这些参数值的变化将如何影响你小车的训练结果。
  - 对比在不同的参数组合下小车的运行结果，并将结果打印出来你（你可以复制 runner.plot_results() 代码对结果多次打印。请至少对比三组参数组合的结果。
- 这样你的报告会更加严谨～

- 在训练中，不同的迷宫将会给训练带来很大的影响，如：
  - 迷宫本身较大或者较难，不利于训练，将会造成训练结果很糟糕。
  - 陷阱位置不好：如下图所示，在通往终点的路上，有一个陷阱，也会导致训练结果变差。

- 因此，建议你固定迷宫，再对比不同参数下的结果；同时你也可以把迷宫看作一个可以修改的变量，来看看如何调整参数，才能来对复杂迷宫学到一个成功策略。

- 关于炸弹堵死道路的问题，你是不是可以：
  - 设置机器人，在遇到陷阱的情况下，增大随机探索的机率，能从陷阱跳出来？
  - 尝试调整 reward 的设置？
【优化】
- alpha 决定了学习的快慢，但总体上说它能够让我们的学习曲线更为平滑，所以它的调整对整个学习过程的影响不是特别大。当然如果你将它置为0，那么 agent 永远无法学习新的策略，学习的策略无法被优化～
- 关于 gamma ，你会发现它对学习的影响会非常大，这是由于它对未来奖励的影响实际上是呈指数级别的，你可以参考 bellman equation 的推导，来深化对它的认识。
- 关于最优参数的组合，实际上考虑几个方面，一个是 alpha 和 gamma 的使用，一般来说它们俩取 0.5 和 0.9 是比较保守的初始值，你可以根据训练的结果来适当增大 alpha 和 gamma 以达到更快的训练速度；对于 epsilon0 及 epsilon 衰减函数，我们会首先尝试 0.5 的初始值以及比较慢的下降函数（`0.99**t`），来尝试训练结果；如果迷宫较为复杂、探索不够，导致无法成功学习策略，则再适当增大随机探索，增大初始值、减小衰减系数。
【思考】
- reward 的设定和能否成功学习到一个策略息息相关。
- 尽管在我们的项目中，我们预先帮你设置好了环境中各个情况的reward。那么对于环境的 reward，你有思考过怎样才能确定一个合理的 reward 值呢？
  - 绝对值应足够大，能够让不好的 action 受到足够惩罚？或者绝对值是不是越大越好呢？
  - 为什么「其余情况」的值，也就是默认的奖励值，是 负0.1？
  - 能否成功地让机器人成功跨越拦在路上的炸弹？


``` python
import random

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.2):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = 0
        else:
            # TODO 2. Update parameters when learning
            '''
            你也可以尝试 cos 函数等衰减函数，你可以这么做：
           if self.alpha <= 0:
           self.alpha = 0
           else:
           self.alpha = 0.5*cos(a * self.t)
           其中 a= math.pi/2/3000。
           我们需要的是 cos 函数在一个单调区间上的单调性，所以可以通过三角函数的技巧使得单调区间符合我们的要求。
            '''
            self.epsilon *= 0.99
            
        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()# 返回当前位置

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        '''
        注意，在 Python 中，0. 和 0 是两个不同类型的数据，前者为 float 类型，后者为 int 类型。那么在这里，我们应该使用前者来对Qtable 进行初始化。
        '''
        if state in self.Qtable: # continue 是跳过， pass 是什么也不做，如果state在Qtable中什么也不做
            pass
        else:
            self.Qtable[state] = {'u': 0.,'d':0.,'l':0.,'r':0.} #如果不存在，创建Qtable[state]，并初始化各个动作的值为0
        
        

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            create_rand = random.uniform(0, 1)
            if create_rand > self.epsilon:
                return False
            else:
                return True

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                return random.choice(self.valid_actions)
            else:
                # TODO 7. Return action with highest q value
                return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        elif self.testing:
            return max(self.Qtable[self.state], key=self.Qtable[self.state].get) #使用max如果不加key，max对比的是key的ascii值，加上key就是key对应的values
        else:
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            
            # TODO 8. When learning, update the q table according
            # to the given rules
            self.Qtable[self.state][action] += self.alpha*(r+self.gamma*max(self.Qtable[next_state].values())-self.Qtable[self.state][action])
            
            

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action
            
        
        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward

```
