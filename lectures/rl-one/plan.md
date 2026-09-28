

**Learning**

Revisit Rosenblatt's learning rule or the supervised learning (to show later contrast with RL)

-------

Slide 2: Deep mind original bet on Artificial General Intelligence



1. Demis Hassabis

Hassabis started with the _problem of intelligence_, not with a particular AI technology

He was a serious chess player extremely young, reaching roughly No. 2 in the world for his age group. At the same time he became fascinated by computers. He has recalled sitting in Foyles reading programming book t about 11, programming an **Othello opponent** on an Amiga because chess was too computationally demanding. The program eventually beat his younger brother

> what is it that my brain is doing when I solve these problems, and could a machine do the same thing?

For Hassabis, **games were miniature universes in which intelligence could be studied**.
this remained almost unchanged as DeepMind's experimental philosophy twenty years later.

I want to build AGI. AI presently doesn't know enough. The brain is the one existing proof that general intelligence is physically possible. I'll spend several years reverse-engineering some of its principles

He wasn't trying literally to copy neurons. His later description is better:

> study neuroscience, identify useful computational principles, and “convert them into an algorithm.”

experiments connecting **dopamine prediction-error signals with temporal-difference reinforcement learning** as psychologically important evidence. If brains actually implemented something akin to RL, then trying to build intelligence around RL was not an obviously crazy idea.


Legg says the cover of their **first investor business plan in 2010** contained one sentence:

> **“Build the world's first artificial general intelligence.”**

Hassabis now paraphrases the plan even more memorably:

> **Step one: solve intelligence. Step two: use it to solve everything else.**

Their 2010 thesis was roughly:

> **Intelligence can be understood algorithmically.**
> 
> General intelligence requires learning rather than hand-programmed behaviour.
> 
> The brain offers useful hints.
> 
> An agent should perceive an environment, learn an internal representation, act, receive feedback and improve.
> 
> Increasing compute will make increasingly general learning algorithms practical.



Slide 3. Deep Q Learning Paper

DeepMind's later DQN paper wasn't just:

> wow, AI plays Breakout.

It was intended as evidence for something much stronger:

> **the same learning algorithm can receive raw sensory data in many different environments and learn what actions maximize reward without being rewritten for each one.**


Suleyman describes DQN as:

> **“really the thing that changed everything for us.”**


onsider what they gave the system.

Not:

> here's the rule of Breakout, here's a representation of the paddle and ball, and here's an algorithm for choosing where to move.

Instead:

**pixels + score/reward + possible controller actions.**

The same architecture was then tried across many games.

And in Breakout something wonderful happened.

Hassabis describes watching the learner progress:

after ~30 minutes, terrible;

after an hour, visibly improving;

after a couple of hours, competent;

and after several hours it discovered the well-known strategy of **digging a tunnel through the bricks and bouncing the ball behind the wall**.

Nobody had explicitly told it that strategy existed.

That was the magic demonstration.

Not because Breakout matters.

Because:

> **the system had gone from raw sensation to representation to action to a non-obvious strategy by learning.**

That looked like a tiny closed-world version of the entire DeepMind thesis.

The first DQN work appeared in 2013; the expanded version became the landmark 2015 _Nature_ paper demonstrating one agent across 49 Atari games.


Slide 4: Googlle Acquisition

The reported purchase prices vary because terms were private—contemporary reports ranged around **£400m / $500m+, with later reconstructions citing roughly $650m**. Google announced the acquisition in January 2014





Slide 5


# And then, almost immediately, DeepMind produces the demonstration that shocks everyone

## October 2015: Fan Hui

AlphaGo defeats professional Go player Fan Hui.

DeepMind keeps the result relatively quiet until publication.

Then:

## January 2016

The _Nature_ AlphaGo paper appears.

## March 2016

AlphaGo plays **Lee Sedol**.

More than 200 million people watch.

AlphaGo wins **4–1**.

And then **Move 37** happens.

Hassabis now describes it as the moment demonstrating that the system could go beyond imitation and find strategies that human experts had not identified.

This happens about **three months after OpenAI publicly launches**.

### AlphaGo Zero — October 2017

Go only.

### AlphaZero — announced late 2017, fuller evaluation published 2018

same broad approach applied to:

- Go;
- chess;
- shogi.

DeepMind describes AlphaZero as one system teaching itself these games from scratch.

For Hassabis and Silver this is the dream:

same algorithm+different environments→superhuman capability.\text{same algorithm} + \text{different environments} \rightarrow \text{superhuman capability}.

This gets much closer to **general-purpose learning**.


#2017: AlphaGo Zero removes the humans

This is probably even more important conceptually than the Lee Sedol victory.

Original AlphaGo uses human professional games.

**AlphaGo Zero does not.**

It receives essentially the rules and plays itself.

Starting from random play:

self-play→experience→policy/value improvement→stronger self-play.\text{self-play} \rightarrow \text{experience} \rightarrow \text{policy/value improvement} \rightarrow \text{stronger self-play}.

And eventually it surpasses the earlier AlphaGo.

DeepMind explicitly frames the significance as escaping dependence on human knowledge.

This is getting close to a recursive motif:

system generates its own training curriculum.\boxed{ \text{system generates its own training curriculum}. }

No wonder people concerned with AGI paid attention.

---

# Then AlphaZero generalizes it beyond Go

There is a naming distinction worth keeping precise:

**Hinton / Brain / sequence models / massive compute**

plus:

**Hassabis / Silver / DeepMind / deep reinforcement learning.**

The race is becoming institutionally visible.

---




------


Slide 6 onwards...now lets look at the invention of RL




What is RL? 

It is an abstraction of.....
Original Psychological motivation


Slide 7
How was it invented? 


Slide 8
David Silver Lecture 1 (youtube transcript)





------

Slide 9
Pong from Pixels

What Karpathy means by “learning Pong from pixels” is that


The agent is **not told what Pong is**.. It is not given variables like: ball position = (83, 42) , ball velocity = down-right  , my paddle position = 97 . opponent paddle position = 51
Instead, at every instant it receives something much closer to what a camera would receive: a huge grid of pixel intensities. In his setup, the raw observation is a `210 × 160 × 3` image, or 100,800 numbers, and the agent has only two choices: move the paddle **UP** or **DOWN**. It gets `+1` when it scores, `-1` when it concedes, and usually `0` for hundreds of intervening actions.
That is amazing because there is an enormous conceptual gap between the input and the thing that must be learned.
Imagine I gave you this repeatedly: [0, 0, 0, 144, 144, 0, 0, 0, ... 100,000 numbers ...] and told you only: Choose A or B. Then occasionally, much later: +1
I never tell you:  that some pixels constitute a ball, that another group constitutes your paddle  that objects persist from frame to frame, that the ball has a velocity, that it bounces. that your paddle can intercept it, that moving up now can cause a reward 20–50 frames later what a “good move” is.

Compare it with the easy version

Suppose you programmed Pong using this input:

```
ball_x = 120
ball_y = 80
ball_velocity_y = +4

paddle_y = 60
```

Then a decent rule is almost obvious:

```
if ball_y > paddle_y:
    move_down()
else:
    move_up()
```

The programmer has already done nearly all the intelligence. You've supplied the machine with the concepts **ball**, **position**, **velocity**, and **paddle**. With pixel RL, you give it: mage → ??? → ??? → UP/DOWN. and reward. The `???` has to emerge through learning.

----
Suppose this happens:

```
t=1     move UP       reward 0
t=2     move UP       reward 0
t=3     move DOWN     reward 0
...
t=47    move UP       reward 0
...
t=81                  reward +1
```

Which action caused the `+1`?

Karpathy highlights exactly this: you're changing perhaps a million neural-network parameters, making thousands of actions, and receiving a meaningful reward only much later. How do you determine which earlier parameter-dependent decisions deserve credit? That's the credit-assignment problem


-----

**Run Code**


Getting Started
- Jupyter Notebook worksheet1: Setup everything
	- Introduction to OpenAI RL Gymnasium
		- What is the observation (sensor input)? Image
		- What is the env (environment)? ALE
		- What is the neural network policy? 


What is the algorithm?
	- 
		- Output: Atatri Pong window playing
- How do we feed in the sensor? As a vector list of numbers
	- Diagram: Show the input.....
	- Preprocess function
		- Colour is not necessary
	- Shall we build a motion information?
		- What is an Inductive bias?
- Defining a neural network in pytorch
	- Policy
- 



Karpathy's original has a 6400→200→1 network,


```
                SHAPE                CONTENT

obs
                210×160×3            RGB pixels
                     │
                     │ [35:195]
                     ▼
obs_cropped
                160×160×3            cropped RGB pixels
                     │
                     │ [::2, ::2, 0]
                     ▼
obs_downsampled
                 80×80               one-channel intensities
                     │
                     │ !=144, !=109, !=0
                     ▼
obs_binary
                 80×80               True / False
                     │
                     │ .flatten()
                     ▼
obs_flattened
                  6400               vector of 0/1-like values
```


```
beautiful Atari screenshot
          ↓
remove irrelevant screen regions
          ↓
throw away 3/4 of spatial pixels
          ↓
throw away 2 colour channels
          ↓
throw away exact pixel intensities
          ↓
retain only object/background
          ↓
turn image into a vector
          ↓
neural network
```

-----
General purpose algorithm:

Policy gradients give an almost comically crude answer:

Try things. If the eventual outcome was unusually good, make the actions you took slightly more probable. If it was bad, make them slightly less probable. Across enormous numbers of trials, the statistical signal accumulates.

The important pattern is: raw sensory data→actions→eventual reward\boxed{\text{raw sensory data} \rightarrow \text{actions} \rightarrow \text{eventual reward}} rather than: human-designed representation→human-written rules→action.


Robotics: detect cup, estimate cup, pose estimate, hand pose calculate, grasp point, calculate trajectory

Instead: camera pixels -> proprioception -> maybe tactile signals ↓ neural network -> motor commands -> did the cup get picked up?




----
Welcome to the era of experience 

David Silver Ineffable Intelligence: 2024


-------
