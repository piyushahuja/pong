


Yes. The broader idea is something like representation design or information engineering:


transform raw observations into a form in which the task-relevant structure is easier to see.

Karpathy’s frame differencing is a beautifully simple case. The raw image contains position; the difference of two images exposes change, and change is a proxy for velocity.

transform raw observations into variables in which the task-relevant structure is easier to see

The raw image contains position; the difference of two images exposes change, and change is a proxy for velocity.

The transform does not create information from nothing. Rather, it can do one of three things: combine multiple observations so previously unavailable information becomes observable; discard nuisance variation; or re-express existing information in coordinates where the relevant relationship becomes much simpler.

A useful general form is

$$
z_t = T(x_t,x_{t-1},\ldots)
$$

where $x$ is raw data and $z$ is the representation actually given to the learner.
1. Differences and derivatives: expose change rather than level
Karpathy uses

$$
I_t-I_{t-1}
$$

which approximates a temporal derivative.
This appears everywhere. In finance, returns

$$
r_t=\log P_t-\log P_{t-1}
$$

are often more useful than raw prices, because the absolute price level is usually less relevant than relative change. In speech recognition, “delta” and “delta-delta” features capture how spectral coefficients change over time. In robotics,

$$
q_t-q_{t-1}
$$

reveals joint velocity if you only observe joint positions.
An event camera takes this philosophy almost literally: rather than repeatedly recording the whole image, each pixel reports changes in brightness. It asks, in effect, “what changed?” rather than “what does the scene currently look like?”
So one general move is:

$$
\boxed{\text{absolute quantity}\rightarrow\text{change in quantity}}
$$

2. Frame stacking: include enough history to make the world observable
Sometimes one difference isn't enough.
A single Pong image cannot tell whether the ball is moving left or right. Two images largely can. Four images can reveal acceleration and resolve ambiguities better.
Hence Atari DQN famously stacked several consecutive frames:

$$
(I_t,I_{t-1},I_{t-2},I_{t-3})
$$

The important idea is not specifically “use four frames.” It is:
if the instantaneous observation does not contain enough information to choose an action, include enough history to recover the hidden state.

In robotics, a camera image may show where an object is but not its velocity. Add previous frames and you can infer motion.
This leads directly to recurrent neural networks, state-space models and belief states. They all try, in progressively more sophisticated ways, to construct

$$
h_t=f(x_1,\ldots,x_t)
$$

where $h_t$ contains what matters about the past.
This is fundamentally an observability problem.
3. Change coordinates so the variables match the problem
Suppose a robot hand is at

$$
(7.2,3.4,1.5)
$$

and the object is at

$$
(7.5,3.2,1.6).
$$

Those absolute coordinates may be less useful than

$$
\text{object}-\text{hand}
=(0.3,-0.2,0.1).
$$

The latter directly says:
the object is 30 cm right, 20 cm back, 10 cm up.

So rather than learning:

$$
f(\text{hand position},\text{object position})
$$

you give the network:

$$
f(\text{object position}-\text{hand position}).
$$

This is common in robotics. Goal-conditioned environments frequently include quantities like

$$
\text{desired goal}-\text{achieved goal}.
$$

You're converting from world coordinates into task-relative coordinates.
The same trick occurs all over science. Classical mechanics often becomes simpler in center-of-mass coordinates. Physics uses relative coordinates and normal modes. Computer vision often uses coordinates relative to an object or camera.
This is arguably one of the deepest forms of feature engineering:
find the coordinate system in which the rule becomes simple.

4. Remove nuisance information
Karpathy's Pong preprocessing also did this.
The raw frame contained:
- background color;
- scoreboard;
- irrelevant pixels;
- RGB information;
- resolution much higher than needed.
He cropped, downsampled, removed background colors and binarized the image.
That sounds crude, but it encodes a powerful prior:
the policy probably needs the paddles and ball, not the decorative pixels.

Formally, suppose

$$
x=(s,n)
$$

where $s$ is task-relevant signal and $n$ is nuisance variation.
You would like a transformation $T$ such that

$$
T(x)\approx T(s)
$$

and varies little with $n$.
Examples include image cropping, background removal, illumination normalization, speech normalization across speakers, subtracting sensor baselines, and removing the gravitational component from an IMU.
This connects directly to invariance.
5. Build invariance into the representation
Suppose the task shouldn't change if an object is translated five centimeters to the right.
Rather than force a neural network to learn that independently for every position, construct a translation-invariant representation.
Likewise for rotations, permutations or scale.
Examples:

$$
d_{ij}=\|x_i-x_j\|
$$

turns coordinates into relative distances and therefore removes absolute translation.
For sets, pooling such as

$$
z=\sum_i \phi(x_i)
$$

removes dependence on item ordering.
For your CycloFormer work, cyclic/permutation structure is basically this same move at the architecture level: declare that some transformations should not alter the underlying meaning.
So representation design and architectural inductive bias lie on a continuum:
preprocess the data
        ↕
choose the coordinates
        ↕
choose architecture symmetry

All three encode assumptions about what information matters.
6. Move into a domain where the hidden structure is sparse
Raw audio is a waveform:

$$
x(t).
$$

But speech recognition historically became easier when transformed into frequencies:

$$
x(t)\xrightarrow{\mathrm{FFT}}X(f).
$$

Even better, use a time-frequency representation such as a spectrogram:

$$
X(t,f).
$$

A waveform that looks impossibly complicated can suddenly display clear harmonics and phonetic structure.
The same happens with EEG, EMG and vibration data.
You can transform:
time series
↓
FFT
↓
frequency spectrum

or
time series
↓
STFT / wavelet transform
↓
time × frequency

The information isn't necessarily new. You've changed the basis so regularities become visible.
This is one of the clearest examples of:
computation can be made easier by choosing the representation in which nature's structure is sparse.

7. Extract phase, amplitude, envelope, etc., instead of using the raw wave
Suppose

$$
x(t)=A(t)\cos(\phi(t)).
$$

Depending on the problem, the useful variables might not be the raw samples at all.
You might want:
- amplitude $A(t)$;
- instantaneous phase $\phi(t)$;
- instantaneous frequency $d\phi/dt$;
- envelope;
- cross-channel phase differences.
For biosignals this can be extremely important.
Again:

$$
\text{raw waveform}
$$

is replaced by physically or functionally meaningful coordinates.
8. Construct sufficient statistics
Sometimes hundreds or millions of observations can be compressed into a few quantities without losing information relevant to the parameter being estimated.
For Gaussian data, for example, mean and variance can contain the important information for particular estimation problems.
Instead of retaining

$$
x_1,\ldots,x_N,
$$

you might work with

$$
\sum_i x_i,\qquad
\sum_i x_i^2.
$$

Your student's federated causal-bandit paper actually uses this philosophy: local data are represented through covariance/sufficient-statistic matrices rather than moving all raw observations around.
This is an especially rigorous version of information engineering:
What is the smallest representation that preserves everything needed for the question I'm asking?

9. Residuals: remove what is predictable and model what remains
Instead of modeling

$$
x_t,
$$

model

$$
x_t-\hat{x}_t.
$$

This appears everywhere: residual networks, time-series forecasting, predictive coding, error-state Kalman filters, control systems.
If most of the signal is boring and predictable, the residual focuses computation on surprise.
You can interpret Karpathy's frame difference partly this way too:

$$
I_t-I_{t-1}
$$

removes everything that stayed constant.
The static background vanishes automatically.
Only change survives.
10. Whiten or normalize so numerical scale doesn't obscure structure
Suppose one feature ranges from

$$
0\text{–}1
$$

and another from

$$
0\text{–}10^6.
$$

A learner may have difficulty purely because of conditioning.
So we transform

$$
x\rightarrow\frac{x-\mu}{\sigma}.
$$

Or whiten:

$$
x\rightarrow\Sigma^{-1/2}(x-\mu).
$$

PCA goes further and rotates the coordinate system into directions of variance.
This isn't usually adding semantic information. It is making the geometry of optimization easier.
That is another important category:
some preprocessing is for the information geometry of the learner, rather than the semantics of the task.

11. Optical flow is a richer version of Karpathy's trick
Frame differencing says roughly:
something changed here.

Optical flow tries to estimate:
this pixel/object moved from here to there.

So instead of two images you construct a vector field

$$
v(x,y)=(v_x,v_y).
$$

For Pong, you might get:
ball:
position = (52, 31)
velocity = (+4, -2)

instead of giving the network thousands of pixels.
You've moved from raw perception toward latent physical variables.
In robotics this is common: estimate object pose, velocity, contact state, affordances, etc., before control.
12. Contact maps are another robotics example
A dexterous hand could receive raw tactile sensors:

$$
s_1,\ldots,s_{92}.
$$

But perhaps what matters is:
thumb contacting object
index contacting object
object slipping clockwise

You can transform raw sensors into a contact map or contact graph.
Now the controller operates on variables closer to the mechanics of manipulation.
This is essentially the same idea as Pong differencing, just further downstream.
There is one especially deep distinction
Some transformations merely make existing information easier to use.
For example,

$$
(x,y)\rightarrow(x-y,x+y)
$$

doesn't necessarily destroy anything: it's just a change of coordinates.
But some transformations intentionally destroy information:
RGB → grayscale

or
image → object position

or
continuous coordinate → category

That can be desirable.
The ideal is often:

$$
\boxed{\text{discard everything irrelevant while retaining everything predictive of the task}}
$$

This is close to the intuition behind the information bottleneck.
If $X$ is raw data, $Z$ the representation and $Y$ what you care about, you want $Z$ to be small/simple while preserving information about $Y$:

$$
I(Z;X)\quad\text{small-ish}
$$

while

$$
I(Z;Y)\quad\text{large}.
$$

And Karpathy's particular trick has another subtlety
He isn't simply compressing a single image.
A single frame fundamentally lacks velocity information.
By computing

$$
I_t-I_{t-1},
$$

he first uses two observations, which jointly contain temporal information, and then compresses them into a representation emphasizing motion.
So there are actually two moves:

$$
(I_{t-1},I_t)
$$

adds history, then

$$
(I_{t-1},I_t)\rightarrow I_t-I_{t-1}
$$

selects the useful temporal feature.
That distinction matters.
A taxonomy I find useful
Most of these techniques can be understood as asking one of five questions:
Question	Typical transformation
What changed?	differences, derivatives, optical flow
What history do I need?	frame stacking, RNN state, Kalman filtering
What should not matter?	normalization, invariance, canonicalization
What coordinate system makes the relationship simple?	relative pose, Fourier basis, log ratios
What is the minimal task-relevant summary?	sufficient statistics, embeddings, state estimation


And then there's a sixth:
What latent variable would make the problem almost trivial if I could recover it?

For Pong, that might be:

$$
(\text{ball position},\text{ball velocity},
\text{paddle position}).
$$

The 100,800 raw RGB numbers are arguably just an awkward sensory encoding of those few variables.
For dexterous manipulation the analogous latent representation might be:

$$
(\text{object pose},\text{object velocity},
\text{finger-object contacts},
\text{slip},
\text{goal-relative pose}).
$$

That is why this issue is much deeper than “preprocessing.” It is really about finding the state representation in which the causal structure needed for prediction or control becomes legible.
And that connects surprisingly closely to your earlier interest in inductive bias: preprocessing, feature engineering, architecture design and scientific abstraction are all versions of the same move—choose what distinctions the system should preserve, and which ones it should ignore.
