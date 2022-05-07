TIPS SHARED IN STUDY GROUP: UNIT 1:


What is the reason for stacking multiple environments? Using the make_vec_env().
From SB3 doc: Vectorized Environments are a method for stacking multiple independent environments into a single environment. Instead of training an RL agent on 1 environment per step, it allows us to train it on n environments per step. Because of this, actions passed to the environment are now a vector (of dimension n). It is the same for observations, rewards and end of episode signals (dones). 

 https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html 


Add !pip install pyglet==1.5.1 before installing gym and it now worked 

![image](https://user-images.githubusercontent.com/82262606/167249993-d17014ec-1558-4d2c-abc7-8e365cc5bebd.png)



Taking small number of steps to solve problem is better. In general, the environment should have reward associated with the number of steps.

![image](https://user-images.githubusercontent.com/82262606/167149937-c7994871-2305-4227-9ea0-6462d647ddfa.png)


just for clarity: a step here is the sequence (state, action)->(next state, reward), right? A “time step” so to speak


![image](https://user-images.githubusercontent.com/82262606/167151324-af4d1870-9931-4c73-8d33-3e7775248d64.png)

You can change this parameter but setting too high can explode your memory

![image](https://user-images.githubusercontent.com/82262606/167151746-ff1a0f58-f921-4fca-ab35-326cd46a08c2.png)


![image](https://user-images.githubusercontent.com/82262606/167152175-17dde98c-cbd4-4046-8c9b-c4c947cbc047.png)


Notebook that show the tradeoff when using multiple envs: "Multiprocessing" in  https://github.com/araffin/rl-tutorial-jnrr19#content

![image](https://user-images.githubusercontent.com/82262606/167153261-3ae1351b-6e6b-46c4-86e1-25c851669504.png)



For  more consistent scores, you might also want to increase the number of evaluation episodes since as you can see PPO can be flaky



![image](https://user-images.githubusercontent.com/82262606/167154676-524cbd39-8725-4d0b-94e2-993dc0538fc2.png)





https://towardsdatascience.com/deep-reinforcement-learning-and-hyperparameter-tuning-df9bf48e4bd2


**SEED:**

You can provide a fixed seed to the environment by adding seed = when you create your environment and your evaluation environment. The variation is not gigantic in your results but yes clearly reproducibility is one of the main challenge in deep RL as we'll see in future units (in addition with sampling efficiency and transfer learning)

eval_env = make_vec_env(env_id,  seed=0, vec_env_cls=DummyVecEnv)


-you probably need to evaluate on more test episode to have a more constant score








---------------





TIPS SHARED BY CEYDA CINAREL:

If you look at the gym environment documentation here: https://www.gymlibrary.ml/environments/box2d/lunar_lander/#rewards
the highest possible score (for v2) is : 
200(solved)+100(at rest)+20(both legs) - (spent fuel)

meaning your model is rewarded to land while spending less fuel. this is why you can usually see a trend of mean episode length getting shorter after your model figures out the landing part (ie after mean reward around 140-170). As you can see in the figure screenshot below (at least this was my observation, I'm no RL expert)

I encourage everyone not to get stuck on the leaderboard aspect of this, only spending gpu hours by doing hyper parameter tuning without gaining a real understanding of the
a) the problem/ environments (what is the action space, what is your state, how does it change etc.)
b) the model / hyperparams
c) understanding the outputs
I mean this is a toy problem, the objective is to learn 🤗 

You can do these by reading documentation & running small experiments (not necessarily long running)

This is my first time doing a hands on RL too! I only knew the very basic concepts of RL covered in this unit (learned briefly at uni in 2017 ancient). So I'm trying to learn a lot by trying to understand the behind the code aspects too. For example what would happen if I changed the action space to not include the main engine, what if two actions were taken at a time, increasing the action space (does that even make sense?) I'm currently looking into wrapper env feature. maybe I'm getting ahead of myself and modifying the env will be covered in future unit.

Read the docs, think & ask questions and have fun







-------------------

WANDB:

https://docs.wandb.ai/guides/integrations/other/stable-baselines-3




-------------------

COLABGYMRENDER:

colabgymrender directly generates a video of your agent and display it in colab


https://github.com/huggingface/huggingface_sb3/blob/main/notebooks/Stable_Baselines_3_and_Hugging_Face_%F0%9F%A4%97_tutorial.ipynb


!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!pip install colabgymrender==1.0.2

import gym
from colabgymrender.recorder import Recorder

env = gym.make('LunarLander-v2')
directory = './video'
env = Recorder(env, directory)

obs = env.reset()
done = False
while not done:
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)

env.play()






---------------
Kaggle:

!sudo apt-get install git-lfs



---------------


AttributeError: module 'gym.envs.box2d' has no attribute 'LunarLander'

!pip install gym[box2d]==0.12  or 0.21???




--------------------

Other Models:

A2C/DQN:

![image](https://user-images.githubusercontent.com/82262606/167250145-f934444c-3a82-4995-8aa8-c94d91d11cd2.png)

A2C: https://huggingface.co/araffin/a2c-LunarLander-v2
DQN: https://huggingface.co/araffin/dqn-LunarLander-v2



-------------------
EVALUATION:
https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading

![image](https://user-images.githubusercontent.com/82262606/167250313-d2448bf2-ed12-4d53-b83e-af4066eabe76.png)




----------------------
TENSORBOARD INTEGRATION:

%load_ext tensorboard
%tensorboard --logdir $tensorboard_log
