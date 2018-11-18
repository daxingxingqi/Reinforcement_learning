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
            #self.epsilon = self.epsilon/(self.t+1)
            #self.t += 1
            你这里的 epsilon 衰减太快了，小车行动5次后，就几乎不随机选择动作了（0.5*(6!)=0），这样会导致小车不能很好地对周围环境进行探索～
请尝试减慢 epsilon 衰减速度（例如每回合以0.99的指数进行衰减），或者使用cos函数作为衰减函数～
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
