# Lecture 2 — Delivery Notes
fin
Companion to `lecture2.html`. Part I is the spoken script, slide by slide. Part II is
sourced background: facts, dates, quotes, and the places where the story is shakier
than it sounds. Read Part II once before the lecture and Part I on the morning.

---

## At a glance

**The argument in one line.** Last week's machines learn what we already know, because
somebody had to write down the answers first. This week's machines learn what nobody
knows, because the only thing they are told is whether it went well.

**The four moments that must land.** If everything else falls apart, get these four.

1. **The AlphaGo card on slide 03** — the theorem promises the function exists; nobody on
   earth can supply its answers. Ten seconds, and it is the cheapest setup in the deck.
2. **The table on slide 06** — for a Go position and a robot arm there is no correct answer
   to supply. Not a limit of our patience: the absence of a teacher, and the reason a
   different kind of learning had to exist.
3. **Move 37, slide 10** — a move no human would have played, produced by a system that
   could not have been shown it. The payoff of the title slide.
4. **Credit assignment, slide 23** — eighty decisions, one number, no indication of which
   decision earned it. The difficulty of the entire subject in one picture.

**Shape and timing.** Two possible shapes, because the slides alone fill ninety minutes and
the notebook will not fit inside that.

*Single 90-minute lecture, notebook in the following session:*

| | | |
|---|---|---|
| Recap, and the turn | 01 – 05 | 11 min |
| Where the contract breaks | 06 | 6 min |
| Why DeepMind bet on this | 07 – 08 | 8 min |
| What RL is: loop, vocabulary, reward hypothesis | 09 – 11 | 11 min |
| Where the idea came from: Thorndike, dopamine | 12 – 13 | 8 min |
| What makes it hard: Silver's four, exploration | 14 – 15 | 6 min |
| What it achieved: DQN, the claim, move 37 | 16 – 18 | 12 min |
| Pong from pixels, and REINFORCE | 19 – 27 | 18 min |
| Limitations, and what comes next | 28 – 30 | 6 min |
| Close | 31 – 34 | 5 min |
| | | **91 min** |

*Two-part session with the notebook in the middle:* run 01 – 28, break for the notebook,
then 29 – 34. Slides 29 and 32 are written assuming they have done the notebook by then.

> **The three history slides now sit together (16 – 18)** and in the order that makes the
> argument: what they handed the machine, what they claimed about it, and what it eventually
> produced. Putting them after the theory means the room already has *state*, *action* and
> *reward*, so DQN can be described properly rather than gestured at — and move 37 lands
> three slides before they go and build the crudest possible version of the same idea.

**If you are running short,** cut in this order: 21 (the easy version), 32 (two bets),
08 (the business plan), 04 (the regression demo — the point survives as a sentence, though
05 needs it).

**Never cut 03, 06, 18, 22, 27 or 29.** 03 sets the whole lecture up in one card; 06 is the
argument; 18 is the payoff; 22 is the difficulty of the subject in one picture; 27 is the
only place they watch something learn; 29 is the only place the lecture admits what this
method cannot do.

**Running callbacks to Lecture 1.** Say these out loud when you reach them, because the
continuity is half the pedagogy.

- the sand question (01) → Lecture 1's opening slide with the number changed
- the AlphaGo card on the UAT grid (03) → "who tells it the right answer?" → 06, again at 12
- *Curve fitting, live* (04) → *now take the dots away* (05). Same demo slot, no answer key.
- `w ← w + η(t − y)x`, and where *t* comes from (06)
- "Where could learning live?" → this week, where could the *signal* come from (06)
- "it's just curve fitting" → same network, different meaning for its output (12)
- the pedestrian/segmentation example → the representation argument (21)
- WaterWorld (05) → *REINFORCE, live* (27) → and back to WaterWorld at the end of the hour
- *A machine can learn to do a thing that nobody knows how to do* → stated for the first
  time under the Go board (18), and once more at (33)

---

# Part I — The script

## Slide 01 · Title — the same question, renumbered

Borrowed verbatim from Lecture 1 with one character changed. *There are artifacts you can
build out of sand that convert energy into intelligence.*

Put it up, say nothing, and let them notice the number. Somebody will.

The line, if you want one: **"Same question, second instalment. Last week we found out
where the intelligence goes. Today we find out where it comes from."**

Then straight into the recap. Do not start explaining here — the whole value of reusing
the slide is that it costs no exposition.

## Slide 02 · Recap — what can these things do

Lecture 1's motivation grid, reused. They have seen it, so do not read it again; that is
not what it is doing here.

Fifteen seconds to re-read, then **one question and only one**:

> *Which of these could a person have written down the right answer for?*

Digits, speech, pedestrians, CT scans — yes, somebody sat down and labelled them. Protein
structures — yes, fifty years of crystallography stands behind AlphaFold. Then the two
that do not belong: **play Go better than anyone alive**, and **move 37**.

Do not resolve it. Plant it, say you are coming back to it in five minutes, move on. This
slide is what makes 1b land, and it costs a minute.

## Slide 03 · Recap — universal approximation

**Do not cut this one.** It is the cheapest setup in the deck.

The theorem in one sentence, and they derived the machinery for it last week: anything you
can write as a function from inputs to outputs, one hidden layer can approximate to
arbitrary precision. Walk two or three of the eight cards — the cards are the point, not
the proof.

Then put the whole weight of the lecture on the sixth card:

> *board state → probability of winning*

And ask it plainly: the theorem promises a network exists that computes that function.
Fine. **Now who is going to tell it what the right answer is?**

For the CT scan there is a radiologist. For the protein there is a crystallographer. For
the winning probability of a Go position there is nobody on earth.

That gap — between a function *existing* and a function being *teachable* — is the whole of
today. Say it once and leave it; 1b cashes it.

## Slide 04 · Recap — curve fitting, live

Two or three minutes, not the fifteen you gave it last week. Today it is a reminder of one
specific thing, and the demo is a prop rather than an exploration.

Run it, let the curve snap onto the points, then point at the points:

> **Every one of those dots is an answer somebody supplied.**

Drag one. The curve follows it — obediently, immediately, to wherever you put it. The
network has no opinion about where the dot ought to be. It has only the dot.

End the recap there, because the next slide takes the dots away.

## Slide 05 · Now take the dots away

**Open this before the lecture starts and leave it running in a tab.** It wants ten or
fifteen minutes of wall-clock time to get good, so start it early and come back to it at the
end — the change over an hour is far more persuasive than anything you can say about it.

Deliver it as the turn, straight after the regression demo, and lean on the contrast: on the
previous slide the curve went wherever you put the dots, obediently and without an opinion.
**Here there are no dots.** Nobody has drawn a correct answer for any frame of this.

Say what it *is* told, because the shortness of the list is the point: red hurts, green
helps. Not where to go, not that the red things move, not that there is such a thing as a
wall. It works out the rest by moving and seeing what happens.

Then read the two lines at the bottom and stop:

> Last week, a machine that learns what we already know, because somebody wrote the answers
> down first. Today, a machine that learns what nobody knows, because the only thing it is
> told is whether it went well.

That is the whole lecture. Everything from here to the end is the machinery underneath it.

> **Housekeeping.** This one is hosted at Stanford and needs the network. If the room's wifi
> is doubtful, load it before you begin and do not reload it. The gridworld on slide 27 is
> local and will always work.

## Slide 06 · Where the contract breaks

**This slide now opens the argument**, since the separate thesis slide has gone. Spend the
first thirty seconds on the contract: put `w ← w + η(t − y)x` back up and ask where *t*
comes from. "The dataset." And where does the dataset come from? *Pause.* A person sat down
and wrote it. Sixty thousand images, sixty thousand human decisions.

Install the vocabulary on the way past — **instructive** versus **evaluative**, the
distinction Sutton and Barto open their textbook on. The version that bites: *I mark your
essay four out of ten. You now know you did badly. You still don't know what a good essay
was, and you don't know which of the twenty things you did was the bad one.*

Then run the table as a poll, row by row, hands up.

Rows one and two: unanimous yes.

Row three is where the room splits and the split is the lesson. Somebody will say "ask a
professional player". Take it seriously, then close it: "Then the best your machine can
ever be is that professional. If the label is a human opinion, the ceiling is human
opinion. And the whole point of the exercise is to get past it."

Row four is the one to linger on, because it isn't about games. Nobody in this room can
write down the joint angles for picking up a cup. Everybody in this room can tell you
whether the cup ended up in the hand.

> **The asymmetry to name explicitly:** impossible to specify, cheap to evaluate. That
> asymmetry is the entire reason reinforcement learning exists.

## Slide 07 · 2010 — games, and the reason for RL

The biography carries two things and nothing else. Do not let it become an anecdote reel.

**What games were for.** Everyone assumes DeepMind did games because games make a good
demo. That is backwards. Games were the laboratory — cheap to run, impossible to argue with
about who won, small enough to do the experiment a million times before lunch. Twenty years
on, that was still how the place ran experiments. The Othello story earns its place only
because it shows the habit starting at eleven.

**Why reinforcement learning.** This is the half that matters and it is the right column.
He did the neuroscience PhD deliberately, as reconnaissance, and one finding came back
mattering more than the rest: dopamine neurons appear to signal prediction error, which is
the quantity TD learning is built out of. So the bet was not that RL was fashionable — in
2010 it emphatically was not — but that the one working example of general intelligence
appears to compute exactly this.

If someone asks whether that means the brain does RL, say no, and mean it. It means one
specific signal behaves quantitatively the way a prediction error would. Narrower claim,
and more impressive for being narrow. You make it properly at 7c.

## Slide 08 · The one-sentence business plan

Read the sentence, then stop for a beat. It is easy to miss now, when every third company
says it, how odd it was to write in 2010. "AGI" got you filed with cranks; the field called
itself machine learning largely to avoid the association.

Then take the five slowly, and do not read them off the screen.

**Four is the one to spend time on**, because it is the only one that tells you what to
build on Monday morning. Perceive, represent, act, get feedback, improve. That is the
agent-environment loop, written as a bullet point, and then fifteen years of implementing
it.

Five deserves a remark too: that is the scaling bet, four years before anybody had the
phrase, and they were right about it in a slightly annoying way.

If a student asks who Shane Legg is — the third founder, writing about machine
superintelligence since his own PhD, and the reason that sentence was on the cover.

> **Moved.** The 2013 DQN slide now sits immediately before Pong from pixels, so that the
> same three things — pixels, a score, the controller — are handed over twice in three
> minutes. Its script is below, out of deck order.

## Slide 09 · What reinforcement learning is

Draw it on the board rather than only showing it. Three arrows; they should be able to
reproduce it from memory for the rest of their lives.

Make the cut explicit, because it is counter-intuitive and exactly right: everything the
agent cannot choose is environment. A robot's motors are environment. Only the
decision-making is agent.

Define nothing yet. Let them look at how little is in the picture. The striking thing
about RL is not its complexity but how few moving parts the formulation has.

## Slide 10 · The vocabulary

Two of the five carry the weight.

**Policy.** "This is where last week connects. The network is unchanged — multiply, add,
squash. What changes is what the output *means* and where the training signal comes from.
Last week: image → label. This week: state → action." If they take one sentence away,
that is it.

**Return, not reward.** This is where sacrifice becomes possible. If the objective were
the next reward, no agent would ever give up a piece in chess. Because the objective is
the discounted sum of everything that follows, the machine can learn that a loss now buys
more later. Every apparently strategic behaviour in this lecture — the Breakout tunnel,
move 37 — is this and nothing more.

Discounting: γ near 1 is far-sighted, near 0 is myopic, Karpathy uses 0.99. One line.

## Slide 11 · The reward hypothesis

Put the challenge at the bottom of the slide to the room and give them ninety seconds in
pairs: **write down the reward function for "be a good doctor".**

Then take proposals and break them, gently:

| They propose | What it actually rewards |
|---|---|
| Cure rate | Refusing hard cases |
| Patient satisfaction | Prescribing what patients ask for |
| Five-year survival | Aggressive treatment of people who'd have been fine |
| Throughput | Six-minute appointments |

Name it: **specification gaming**. It is documented, it has a long public list of real
examples, and it is the same phenomenon as an engagement-maximising feed.

Leave them with the honest position: the hypothesis is not obviously true, not obviously
false, and the difficulty of writing reward functions for things we actually care about
is one of the central open problems in AI safety. It is a real frontier and they can work
on it.

> **The invention section runs in a deliberate order and the order is the argument:**
> 6d behaviour (1898) → 7c the algorithm returns to biology (1997) → 7d the whole chart.
> Three slides, not five. Do not shuffle them.

## Slide 12 · 1898 — Thorndike

The best teaching object in the lecture, because every claim in the formalism is visible
in the animal. Walk the three bullets and point back at the cat for each.

Dwell on **exploration**, because it is a design constraint rather than an observation:
a policy that has stopped trying new things cannot improve — ever. Not because the
algorithm is weak but because the information does not exist.

One caution, if the room is strong: the analogy to real animals is genuinely useful and
genuinely limited. RL is not a theory of how animals learn. It borrowed a principle and
went off to become mathematics.

## Slide 13 · 1997 — the brain computes δ

**This slide now has to define δ**, because the temporal-difference slide has gone. Keep
that to one sentence and do not be tempted into teaching TD: it is a value method, today's
notebook is a policy method, and the detour costs you five minutes you do not have.

The sentence: *Sutton needed a term for "what I expected, minus what I now expect given
what just happened". He wrote it down in 1988, and it made his learning rule converge.*
Then point at the formula, say it is just a number, and move to the monkeys.

Deliver the three conditions as three experiments, and **ask the room to predict the third
before you reveal it.** Most people guess "nothing happens". The dip below baseline is the
detail that convinces, because a pure reward signal has no reason to go negative when
nothing occurs. Something in there is holding an expectation and subtracting.

Then be precise about what this establishes. It does not show the brain runs TD learning.
It shows one specific signal behaves, quantitatively, the way a prediction error would.
Much narrower, and much more impressive for being narrow.

Close the loop with slide 9: this is the result Hassabis points to when explaining why
DeepMind was built on reinforcement learning rather than anything else.

## Slide 14 · Four things that make it different

Keep Silver's order. The fourth is the one nobody anticipates and it belongs at the end.

Make the fourth concrete: "MNIST is MNIST whatever your network does. But a policy that
never moves left never sees what's on the left, so it can never learn that the left was
better. The model's incompetence censors its own training data. There is no analogue of
this in anything you learned last week."

Point them at Silver's lecture series here, not at the end when they're packing up.

## Slide 15 · Exploration and exploitation

Personal before technical. "Hands up: last time you ate at a restaurant you like, did you
order something new?" The room laughs and the intuition is free.

Two technical additions:

- Exploit-only means a policy can be permanently stuck in a mediocre habit and never find
  out, because the evidence that would correct it is exactly the evidence it declines to
  gather.
- The balance should shift over a lifetime: explore when ignorant, exploit when
  experienced. That is a claim about people as much as algorithms, and they will notice.

Payoff: move 37 exists because the system spent an enormous share of its training doing
things no sensible player would try.

## Slide 16 · 2013 — nobody told it there was a ball

> **This opens the last stretch of history**, and it is the first of three slides that
> belong together: **what they handed it → what they claimed → what it produced.** They have
> the whole vocabulary by now, so *state*, *action* and *reward* mean something here — which
> is exactly why this section sits after the theory rather than before it.

Do the absences first. The list of what it was *not* given is far more striking than the
list of what it was. It never sees the number 83 meaning `ball_x`. It sees brightness
values, and the fact that some of them hang together as a ball is itself something it has
to work out.

Then walk the four lines of the Breakout timeline. The first three are good engineering and
nothing more. **The fourth is a different kind of claim**, and the room should feel the gear
change: the tunnel is a genuinely non-obvious idea that strong human players took a while
to find, and it turned up in a system that had never been told what a brick was.

Suleyman's line is worth quoting exactly: *really the thing that changed everything for us.*

Housekeeping: the tunnel clip goes on the next slide. If it hasn't loaded, describe it and
move on — don't stand there fighting a video.

## Slide 17 · The paper was not about Breakout

This is the slide that earns the lecture its title, so give it room.

Draw the contrast with Deep Blue, which they have all heard of. Deep Blue beat Kasparov in
1997 and could not play draughts, because every single thing it knew about chess was put
there by people. DQN plays forty-nine games with the same code because nothing it knows
about any of them was put there by people.

The three cards are the structure — walk *same*, *raw*, *many* in order and let each one
land before the next.

If someone asks what the network actually predicts: good question, hold it. Q-learning is a
value method and we teach policy methods today, through Pong. Give them the shape and
promise the rest — it learns to predict, for each button, how much total score pressing it
now is eventually worth.

**Be honest about the limits** or a sharp student will do it for you. Enormously
sample-hungry. Bad at anything needing long-horizon planning, Montezuma's Revenge being the
standing example. And the accurate figure is *above human level on about half*, not "beat
humans at Atari".

## Slide 18 · Move 37

The emotional centre of the first half. Give it room and do not talk over it.

**If you have the documentary clip, this is where it goes** — around 49:00, ninety seconds
either side. It does more than any diagram will, and it is the cheapest minute in the
lecture.

Set the scene in two sentences and stop: March 2016, Seoul, more than two hundred million
people watching, AlphaGo won the match four games to one. (The earlier Fan Hui match and
the *Nature* paper no longer have a slide — mention them only if somebody asks how it got
there.)

Then make the technical point explicit, because the drama on its own teaches nothing:

> AlphaGo's first version *did* learn from human games. But supervised learning on human
> games can only ever reproduce the statistics of human play, and in human play this move
> occurs about once in ten thousand times. **The supervised part could not have produced
> it.** What produced it was self-play — the system playing itself millions of times and
> keeping what won.

That is the whole distinction of this lecture, compressed into one stone. Say it once,
cleanly, and let it sit.

Then put the title sentence back on the screen and let the room make the join themselves.
It returns a third and final time at 10b.

**Counterweight**, worth thirty seconds because it keeps the story honest and the room
warmer: game four, Lee Sedol's move 78, the "divine move", which AlphaGo had no answer to.
He won that game.

> **What you lose by cutting the Zero slide.** Self-play with *no human games at all* was
> AlphaGo Zero, 2017, and it is the cleanest proof that the human data was a shortcut
> rather than a foundation. That slide is gone, so the one line above is now carrying it.
> If a student asks whether AlphaGo needed human games, the answer is: the first one used
> them, the next one did not and was stronger.

## Slide 19 · Pong from pixels

Slow right down. Four or five minutes on this slide alone.

Read the right-hand column aloud, line by line. Each line is something the room assumed
the machine had. Most students do not realise until they see the list that *"there is a
ball"* is itself learned.

The number to hold: **100,800 in, 2 out.** Everything between is the learning problem.

## Slide 20 · Put yourself in its position

Deliver it straight, then be quiet for five full seconds. The silence does the work.

If you want to push: "How long do you think you'd need before you even *suspected* that
some of those numbers were an object that moved? That's what the network is doing in its
first thousand games."

## Slide 21 · How you would have written it

The pivot of the argument, and it connects straight back to the pedestrian example from
Lecture 1.

"Everyone in this room can write the four-line rule. Nobody in this room can write the
function from a hundred thousand pixels to those four numbers. And *that* is what the
network learns on the way to learning the policy. The features aren't given. They're a
by-product."

One sentence of honesty: for Pong you obviously *would* use the four numbers if somebody
handed them to you. The point is the cases where nobody can — cup-picking, driving,
surgery, conversation.

## Slide 22 · Credit assignment

**Ask them how they'd solve it before you tell them anything.** You will get three
answers and all three are real research programmes:

| What they say | What it's called |
|---|---|
| "Reward the most recent actions most" | Discounting |
| "Work out what would have happened otherwise" | Baselines, advantage, counterfactuals |
| "Learn to predict value at every step so you never wait" | TD — the δ from slide 13. Name it, say it is what DQN runs on, and say we are not doing it today. |

Then the contrast in one sentence: with backpropagation the error is available at the
output on every single example, so credit assignment through the layers is a calculus
exercise. Here it is one number, arriving late, for an entire episode.

## Slide 23 · The preprocessing pipeline

Two questions to the room, in this order.

**"Why throw away colour?"** Because it carries no information about this task. Good — so
somebody looked at the problem and decided that. That is a human prior, quietly smuggled
into a story about learning from raw pixels. Be honest about it.

**"A single frame can't tell you which way the ball is going. So how does it know?"** The
answer in Karpathy's code: the input is the *difference between consecutive frames*. That
difference image is motion, handed over for free. It is the clearest example of an
inductive bias you will find. Ask what else they'd hand over if they were designing it.

Also worth saying, because they assume everything needs a data centre: 6400 → 200 → 1 is
a *tiny* network, and it learns Pong.

## Slide 24 · The network itself

Deflate it deliberately. They arrive expecting the architecture to be where the cleverness
lives, and here it is a first-week exercise: 6400 → 200 → 1, one hidden layer, no
convolutions, smaller than anything they built last week.

Everything hard about this problem is in **where the gradient comes from**, not in the shape
of the net. Say that sentence and let it sit.

Point at the flattening again: turning the 80×80 image into a 6400-vector destroys the fact
that adjacent pixels are adjacent, and the network has to rediscover it. A convolutional net
would not have to. Karpathy uses a plain one on purpose, to show the crudest possible thing
still works.

## Slide 25 · The REINFORCE algorithm

The cancellation argument is the one insight here, and it repays saying twice in
different words:

"An action with nothing to do with the outcome shows up in good episodes and bad episodes
at about the same rate, so over many episodes it gets pushed up as often as down. An
action that genuinely tends to precede winning gets pushed up more often than down.
Nobody assigns the credit. The statistics do, slowly."

That also explains sample inefficiency, which is the honest headline criticism of the
field. A human learns Pong in a minute; this takes days. Ask why and let them speculate —
the answers involve prior knowledge of objects and physics that the network builds from
nothing, and it is an open question, not a settled one.

## Slide 26 · Where the gradient comes from

Put the two diagrams up together and say almost nothing. The visual point is that the
machinery is identical — same network, same forward pass, same backward pass. The only
difference is the number the gradient gets multiplied by.

Upstairs that number is **1**, and it arrives instantly, because a person wrote the answer
down. Downstairs it is the **return**, it may be negative, and it does not arrive for
another eighty frames.

That is the whole lecture in two pictures. If a student has followed nothing else, this is
the slide to send them away with.

Name the consequence: because the downstairs multiplier is noisy, you need enormously more
samples to average the noise out. **Sample inefficiency is not a bug in the implementation.
It is the price of not having a teacher.**

## Slide 27 · REINFORCE, live

The counterpart to slide 04, and the only place in the lecture they watch something learn.
Twenty seconds of silence, then two or three minutes of commentary over the top.

**Before pressing RUN**, make them look at the arrows: four per cell, all the same length. A
policy with no opinion about anything. Nobody has told it where the goal is, that falling in
is bad, or that the walls are walls.

**Press ONE EPISODE a few times first.** It flails. Point at the trajectory — green if it
reached the goal, red if it fell in — and say the line from slide 25: every action on a green
line just became slightly more likely, every action on a red line slightly less, *including
the irrelevant ones*. Nobody decided which mattered.

**Then RUN.** The arrows lengthen and commit, the average-return line climbs, the success
rate goes from nothing to near a hundred per cent. The observation that matters: *at no point
did anyone supply a correct action for any square.* Those arrows are entirely the residue of
outcomes.

Two things to point out while it runs. It learns to avoid the pit long before it finds the
fast route — bad news travels faster, because there is more of it. And if it finds a mediocre
path early it can sit on it, which is exploitation beating exploration, happening live.

> **Say the caveat out loud.** This is a sixty-four-square table, not a neural network. It
> converges in seconds because the state space is tiny. Pong has a hundred thousand inputs
> and takes days. The algorithm is identical; the difference is how much there is to search.

## Slide 28 · Outside the arcade

Land the generalisation, because until now every example has been a game and a sceptical
student is entitled to say so.

**Be fair to the classical pipeline:** it is how working robots are actually built, it is
inspectable, it is debuggable, and when it fails you know which stage failed. The real
argument against it is that it requires somebody to know how to specify each stage, and
for most interesting tasks nobody does.

Where the field has gone since: the strongest current robotics results are hybrids, and
large pre-trained models increasingly supply the prior knowledge that pure pixel RL had
to learn from scratch. That is the frontier, and that is where the interesting jobs are.

## Slide 29 · Four things wrong with all of this

**Do not skip this to save time.** Everything they have seen for eighty minutes has worked,
and a room that leaves believing reinforcement learning always works has been mistaught.

One and three they have already felt. **Two is the one to spend time on**, because it is the
deepest: the CoastRunners boat that scored more by spinning in circles collecting pickups
than by finishing the race. It was not broken — it maximised exactly what it was given. If
you have thirty seconds of the clip, play it; the laugh is worth more than the argument. Then
connect it back to slide 13, where you asked them to write the reward function for *be a good
doctor*. This is what happens when somebody tries.

**Four is the one they will not have thought of**, and it is why this is not in the world the
way the headlines suggest. The method needs a world in which failure is free, and the
interesting problems are all in worlds where it is not.

If the room is strong, plant the follow-up: every one of these four is somebody's current
research programme, and three of them are unsolved.

## Slide 30 · Everything after today

Ninety seconds. A map, not a lesson — they cannot learn any of it from a table, and that is
not what it is for. What it is for is that a student who wants to go further leaves knowing
five real search terms.

Take the **first row** properly, because it is the direct continuation of what they just
coded: add a value network to REINFORCE and you have actor–critic; that is where δ from
slide 15 reappears; PPO is that with a clipped update. Everything modern is in that row.

Then point at the **last row** and stop, because it is the hinge into the close. RLHF
replaced the value function with a human rater. It works spectacularly. And a human rater
cannot score a move that nobody understands — so the ceiling is back. That is the argument of
slide 34, and you have just set it up.

## Slide 31 · Where we are

The summary slide. **Do not read it** — they can read. Say the top line, give them fifteen
seconds with the two columns, then talk to the paragraph at the bottom, which is the part
worth their attention.

The one thing to land: today's four difficulties are not a list, they are **one substitution
seen from four sides.**

- Credit assignment exists because the number is *late*.
- Exploration exists because the agent's own choices decide what data it ever sees.
- Reward hacking exists because a human had to *write the objective down*.
- Sample inefficiency exists because a verdict carries so much less information than an answer.

If they leave with the list, they have memorised something. If they leave with the
substitution, they can derive the list.

Name the trade in the two columns explicitly, because students want one method to be the
better one: **a strong signal buys speed and buys you a ceiling; a weak signal costs speed
and removes the ceiling.** Neither is right in general, which is exactly why the last slides
are about combining them.

The closing sentence can be said flatly. It has earned itself by now.

## Slide 32 · Two bets

A short orientation slide, and it exists because students arrive believing AI means
"train a big model on the internet". They should see that the other tradition existed,
produced the most spectacular results of the decade, and hasn't gone away.

The last line is the payoff, said plainly: RLHF and everything after it is the second
column applied to the first. The reward model is a learned stand-in for a human verdict.
The thing they use every day is a hybrid.

**Close the half:** "We've spent twenty-five minutes on what reinforcement learning *did*.
Nobody has yet told you what it *is*."

## Slide 33 · 2025 — the argument, and the bet

**The two 2025 slides are now one**, with a screenshot of each thing on it. Five minutes,
two halves. Do the left first and do not linger.

**Left — April 2025, the paper.** Silver and Sutton write this lecture down as a
prediction. The three eras are the structure: simulation gave superhuman and narrow; human
data gave general and capped; experience is the claim you can have both. The prosaic reason
for the shift is that the good human data is nearly used up. The real reason is that new
theorems are outside the corpus by definition.

**The ceiling quote is the sentence to read aloud.** Then say what makes it sharp: it is an
attack on RLHF from the RL side. RLHF replaced the value function with a human rater, and a
human rater cannot score a move nobody understands. That is move 37 restated as a design
constraint, and it is the honest reason a chat model does not produce one.

AlphaProof is the one-line example if you want it: a hundred thousand human proofs in, a
hundred million self-generated out, silver-medal standard at the IMO.

**Right — six months later, the company.** He left DeepMind and raised $1.1bn on it, with
no product. Give the facts in one breath and spend your time on the name instead.
*Ineffable* is a word about the limits of language, adopted as a prediction. Everything in
Lecture 1 was learned from things humans had already written down; move 37 was the first
hint that the interesting part might not be sayable at all.

**Then be straight with them**, because they will have read the headlines. The paper is a
position paper, not a result. The company has no product and no paper. And the known
weakness is the one from slide 28, which the paper itself concedes: the method has never made the
jump from closed problems with an exact reward to open problems without one.

Put the title sentence back up and let the room make the join. Do not explain it — it is the
third time they have seen it, after move 37.

Last line, if you want one: *most of the people who will answer that question are currently
about your age.*

## Slide 34 · Then we run it, and contact

Thirty seconds each. Slide 11 is a signpost, not a section — tell them the notebook will
be shared and that the only thing they need to bring is a willingness to watch a number go
up very slowly. Slide 12: say what you are interested in and let them come to you.

---

# Part II — Background

Facts, dates, quotes and citations, with an honest marking of how solid each is.
**[solid]** = documented in a paper or a primary source. **[reported]** = widely
repeated in interviews and press, and worth saying with a hedge such as "he has
described" or "reportedly". **[check]** = repeat only if you verify it first.

## The people and the bet

**DeepMind** founded 2010 in London by Demis Hassabis, Shane Legg and Mustafa Suleyman.
**[solid]**

**Hassabis's chess.** Reached master standard young and was among the very highest-rated
players in the world for his age group as a child; captained England junior teams.
The precise "number two in the world for his age" figure is **[reported]** — say "one of
the highest-rated juniors in the world" and you are safe.

**The Othello program.** He has described, in several interviews, buying a computer with
chess winnings as a boy, reading programming books, and writing an Othello-playing
program that beat his younger brother. Chess was too computationally demanding for the
machine; Othello was not. The specific machine varies between tellings — **[reported]**,
so say "a home computer" rather than naming one.

**The neuroscience.** PhD in cognitive neuroscience at UCL, completed 2009, working on
memory and imagination — including the well-known result that patients with hippocampal
damage struggle to *imagine* new experiences, not merely to remember old ones. He has
consistently framed this as deliberate reconnaissance for building AI. **[solid]**

His own phrasing for the method: study neuroscience, identify useful computational
principles, and convert them into an algorithm. **[reported]**

**The business plan.** Legg's account is that the cover of the 2010 investor plan carried
the line *"Build the world's first artificial general intelligence."* **[reported]**
Hassabis's later paraphrase — *step one, solve intelligence; step two, use it to solve
everything else* — is one he has given publicly many times. **[solid enough to quote]**

## DQN

- **2013** — *Playing Atari with Deep Reinforcement Learning*, Mnih et al., NIPS Deep
  Learning workshop. Seven games. **[solid]**
- **February 2015** — *Human-level control through deep reinforcement learning*, Mnih et
  al., **Nature** 518, 529–533. One agent, one architecture, one set of
  hyperparameters, **49 Atari games**; performance at or above a professional human
  games tester on about half of them. **[solid]**

**What the agent receives.** Raw pixels (downscaled, grayscaled, four frames stacked so
motion is visible), the score as reward, and the set of joystick actions. **[solid]**

**The Breakout tunnel.** The agent discovers that digging through the side of the wall
and putting the ball behind it is far more efficient than chipping away at the front. It
appears in DeepMind's own demonstration video and is the single most-shown clip in the
company's history. The hour-by-hour progression ("thirty minutes hopeless, an hour
better, two hours competent, several hours and it found the tunnel") is Hassabis's own
narration in talks — **[reported]**, and safe to attribute to him rather than to the
paper.

**Suleyman on DQN:** *"really the thing that changed everything for us."* **[reported]**

**Known weaknesses, worth volunteering.** Enormously sample-inefficient (tens of millions
of frames — many weeks of human-equivalent play). Poor on games requiring long-horizon
planning or exploration, with *Montezuma's Revenge* the standard failure case. Two key
engineering tricks make it work at all: **experience replay** (store transitions and
re-sample them, which breaks the correlation between consecutive frames) and a **target
network** (a frozen copy of the network used to compute the learning target, which stops
the bootstrapping from chasing itself). Both are worth one sentence if a strong student
asks why it wasn't done earlier. **[solid]**

## The acquisition — *slide cut, kept here for reference*

The deck no longer has an acquisition slide. This stays because a student may ask how a
four-year-old company with no product got bought, and because the scroll-view essay still
tells the story.

Google announced the acquisition in **January 2014**. Terms were not disclosed.
Contemporary reporting put it at around **£400m / $500m+**; later reconstructions cite
roughly **$650m**. Both figures are defensible — give the range, not a single number.
**[reported]**

An **ethics and safety board** was a reported condition of the sale. **[reported]**

Context worth having: Facebook was also in conversations with DeepMind around the same
period. **[reported]**

## AlphaGo

| When | What | Confidence |
|---|---|---|
| Oct 2015 | Beats **Fan Hui** (2-dan professional) 5–0 in formal games — first program to beat a professional on a full 19×19 board with no handicap. Kept quiet until publication. | **[solid]** |
| 28 Jan 2016 | *Mastering the game of Go with deep neural networks and tree search*, Silver et al., **Nature** 529, 484–489. | **[solid]** |
| 9–15 Mar 2016 | **Lee Sedol**, Seoul. AlphaGo wins **4–1**. Viewership widely reported as 200 million+ (some reports say 280 million). | **[solid]** / viewership **[reported]** |
| Game 2, move 37 | The shoulder hit on the fifth line. | see below |
| Game 4, move 78 | Lee Sedol's "divine move" (신의 한 수); AlphaGo's play collapses afterwards and he wins the game. | **[solid]** |
| Oct 2017 | *Mastering the game of Go without human knowledge* (**AlphaGo Zero**), Nature 550. No human games; from random play it surpasses the Lee Sedol version in **three days**. | **[solid]** |
| Dec 2017 / Dec 2018 | **AlphaZero** — arXiv preprint 2017, full evaluation in **Science** 362 (Dec 2018). Go, chess and shogi from self-play alone. | **[solid]** |

**On move 37 specifically.** AlphaGo's own policy network estimated the probability that
a human player would have chosen it at roughly **1 in 10,000**. That figure comes from
DeepMind and appears in the documentary and in Silver's talks — **[reported but well
sourced]**. Fan Hui's reaction ("It's not a human move — I've never seen a human play
this move") and the commentators' assumption of a bug are both on record. Lee Sedol
stood up and left the table for a period before replying; the often-repeated "fifteen
minutes" is approximate — **[reported]**.

**The technical point that matters more than the anecdote.** AlphaGo combined:
(i) a **policy network** trained initially on ~30 million positions from human games;
(ii) further **policy improvement by self-play reinforcement learning**; (iii) a **value
network** trained on self-play games; and (iv) **Monte Carlo tree search** at play time.
The supervised component can only reproduce the statistics of human play. The move 37
class of result comes from the self-play component. **[solid]**

**AlphaGo Zero's simplifications**, if asked: no human data, one network with two heads
(policy and value) instead of two networks, and search used inside the training loop as a
policy-improvement operator. **[solid]**

**The joseki detail.** The Zero paper and DeepMind's commentary describe it rediscovering
standard human opening sequences, then departing from several of them in favour of its
own. **[solid]**

## The invention of RL

**Thorndike.** *Animal Intelligence* (1898 dissertation; 1911 book). Puzzle boxes — cats
escape by tripping a latch, first by accident, then faster on repeated trials. The
learning curves are smooth and gradual, which was his argument *against* explanations by
insight or reasoning. **Law of effect**: responses followed by satisfaction become more
likely in that situation; responses followed by discomfort become less likely.
**[solid]**

**Skinner.** *The Behavior of Organisms* (1938). Operant conditioning, the term
**reinforcement**, and **shaping** — building elaborate behaviour by rewarding successive
approximations. **[solid]**

**Michie.** *MENACE* (1961), a tic-tac-toe player built from ~300 matchboxes, each holding
coloured beads for the available moves; after a win you add beads to the moves played,
after a loss you remove them. It is reinforcement learning implemented in cardboard and
it is the best physical demo in existence for this lecture — **[solid]**, and see
Appendix D.

**Bellman.** *Dynamic Programming* (1957); Markov decision processes; the Bellman
equation, *V(s) = r + γV(s′)* in its simplest reading. Also coined "curse of
dimensionality". Requires a full model of the environment's dynamics. **[solid]**

**Samuel.** *Some Studies in Machine Learning Using the Game of Checkers* (IBM Journal,
1959). A program that improved by playing, using a form of temporal-difference update
before the name existed. **[solid]**

**Barto, Sutton & Anderson (1983).** The pole-balancing paper — an actor–critic system
learning to balance a cart-pole from nothing but a failure signal. Good concrete example
if someone wants one. **[solid]**

**Sutton (1988).** *Learning to Predict by the Methods of Temporal Differences*, Machine
Learning 3. The TD(λ) paper. **[solid]**

**Watkins (1989).** Q-learning, in his thesis. Worth one sentence only, since DQN is the
Q in Deep Q-Network. **[solid]**

**Tesauro (1992–95).** TD-Gammon — a backgammon program trained by self-play TD that
reached world-class play and changed how human experts played certain opening rolls. It
is the direct ancestor of the AlphaGo story and a good answer to "was AlphaGo the first
time this happened?" **[solid]**

**Sutton & Barto**, *Reinforcement Learning: An Introduction* (1998; 2nd ed. 2018). Free
online at incompleteideas.net. Chapter 1 contains the field's own history of exactly this
lineage. **[solid]**

## Dopamine

**Schultz, Dayan & Montague (1997)**, *A Neural Substrate of Prediction and Reward*,
**Science** 275, 1593–1599. **[solid]**

The three conditions, which is what you actually present:

1. **Unexpected reward** → phasic burst of dopamine firing at reward delivery.
2. **Learned cue then reward** → the burst migrates to the *cue*; the reward itself
   elicits no response, because it is predicted.
3. **Learned cue, reward omitted** → firing *dips below baseline* at precisely the time
   the reward was expected.

This is exactly the behaviour of a TD error term. The claim to make is narrow: one
signal behaves quantitatively like a prediction error. It is **not** a demonstration
that the brain implements TD learning, and the literature since has complicated the
picture considerably (dopamine also carries information about novelty, salience and
movement; the **distributional RL** work from DeepMind in 2020 argued the population
encodes a whole distribution of predictions rather than a single mean). **[solid]**

## Silver's framing

**RL Course by David Silver** (UCL, 2015), Lecture 1: *Introduction to Reinforcement
Learning*. On YouTube via the Google DeepMind channel —
<https://www.youtube.com/watch?v=2pWv7GOvuf0>. Slides at davidsilver.uk.

The items taken from that lecture and used on slide 14:

- **"Reinforcement learning is the science of decision making."**
- **The reward hypothesis:** *all goals can be described by the maximisation of expected
  cumulative reward.* He explicitly calls it a hypothesis.
- **What distinguishes RL from other machine learning:** there is no supervisor, only a
  reward signal; feedback is delayed, not instantaneous; time really matters (the data is
  sequential and non-i.i.d.); and the agent's actions affect the subsequent data it
  receives.
- His reward examples — helicopter (+1 trajectory, −100 crash), backgammon/games (+/−1 on
  the result), robot walking (+1 per metre, −100 for falling), portfolio management
  (reward in currency) — are lifted directly; they are his, and they are good.

Also from the same lecture, if you want more: the distinction between **prediction**
(how good is this policy?) and **control** (find the best policy), and the distinction
between **model-free** and **model-based** methods.

**"Welcome to the Era of Experience"** — Silver & Sutton, 2025. The argument that the
next phase of AI will be driven by data an agent generates through its own interaction
rather than by the finite corpus of human text. It is the natural closing pointer for
this lecture and a good bridge to the next one. **[solid]** — note that the *"Ineffable
Intelligence, 2024"* reference in the original plan does not match anything I could
confirm; **[check]** it before citing, or use the Era of Experience essay instead.

## Karpathy, Pong from Pixels

*Deep Reinforcement Learning: Pong from Pixels*, 31 May 2016 —
<http://karpathy.github.io/2016/05/31/rl/>. **[solid]**

The specifics, all from the post:

- Raw Atari frame: **210 × 160 × 3** = 100,800 numbers.
- Preprocessing: crop rows `[35:195]`, take every second pixel `[::2, ::2, 0]` (one
  colour channel), set background values (144, 109, 0) to zero and everything else to 1,
  then flatten → **6,400**-dimensional vector.
- **The network input is the difference between the current and previous preprocessed
  frames.** This is where motion comes from; without it a single frame is ambiguous.
- Network: **6400 → 200 → 1**, one hidden layer, ReLU hidden units, sigmoid output read
  as P(action = UP).
- Training: REINFORCE-style policy gradients, discount γ = 0.99, RMSProp, batches of
  episodes. Roughly **three nights** of training on one machine to beat the built-in
  opponent reliably.
- His own framing of the difficulty: you are changing a million parameters on the basis
  of a reward that arrives long after the actions that caused it — the **credit
  assignment problem**.
- His summary of policy gradients, which the slide paraphrases: play, and if the outcome
  was good make everything you did more likely; if bad, less likely.

His caveat, worth repeating for honesty: he describes the sample inefficiency plainly and
notes that a human plays Pong well within a minute, which he treats as evidence that
humans bring an enormous amount of prior structure — objects, physics, the notion of
*control* — that the network has to build from nothing.

## The era of experience, and Ineffable

**The paper.** *Welcome to the Era of Experience*, David Silver and Richard S. Sutton,
April 2025. A preprint of a chapter for the MIT Press book *Designing an Intelligence*.
PDF at
<https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf>
**[solid]** — but say *position paper*, not *result*, because that is what it is.

**Quotes worth using verbatim** — all **[solid]**, all from that PDF:

- Abstract: "A new generation of agents will acquire superhuman capabilities by learning
  predominantly from experience."
- On the ceiling: human-centric RL "has imposed a new ceiling on the agent's performance:
  agents cannot go beyond existing human knowledge."
- On what was lost: "something was lost in this transition: an agent's ability to
  self-discover its own knowledge."
- The four dimensions, near-verbatim: agents will inhabit *streams* of experience rather
  than short snippets; actions and observations *richly grounded in the environment*
  rather than human dialogue alone; rewards *grounded in their experience of the
  environment*, rather than coming from *human prejudgement*; and they will *plan and
  reason about experience*, rather than reasoning solely in human terms.

**Three eras.** The paper's own framing is era of simulation → era of human data → era of
experience. Its charge against the simulation era is precise and worth repeating, because
it is the fair criticism of everything on slides 3–5: those agents "did not leap the gap
between simulation (closed problems with singular, precisely defined rewards) to reality
(open-ended problems with a plurality of seemingly ill-defined rewards)." **[solid]**

**The data-exhaustion claim.** "The majority of high-quality data sources — those that can
actually improve a strong agent's performance — have either already been, or soon will be
consumed." **[solid]** as a quotation; **[reported]** as a fact about the world — it is
contested, and a student may well have read a rebuttal.

**AlphaProof.** First program to achieve a medal standard at the International Mathematical
Olympiad. Initially exposed to "around a hundred thousand formal proofs, created over many
years by human mathematicians", then generated "a hundred million more through continual
interaction with a formal proving system". **[solid]** — the figures are the paper's own.

**The DeepSeek line**, if you want a second example that isn't a game: the paper quotes it
approvingly — "rather than explicitly teaching the model on how to solve a problem, we
simply provide it with the right incentives, and it autonomously develops advanced
problem-solving strategies." **[solid]** as a quotation of a quotation.

**The companion podcast.** *Is Human Data Enough?*, Google DeepMind: The Podcast, Hannah
Fry with David Silver, April 2025 — covers the era of experience, AlphaZero, move 37,
RLHF and AlphaProof in about forty minutes. <https://deepmind.google/the-podcast/>
The best single thing to set as listening after this lecture. **[solid]**

---

**Ineffable Intelligence.** Silver founded it in November 2025 and left Google DeepMind in
January 2026, after twelve years there full-time. In April 2026 the company announced a
$1.1bn seed round at a $5.1bn valuation — reported as the largest seed round in Europe —
with Sequoia, Lightspeed, Index Ventures, Google, Nvidia, the British Business Bank and
Sovereign AI among the investors. He remains a professor at UCL. **[solid]** for the dates
and his roles; **[reported]** for the investor list and the "largest in Europe" framing,
which comes from the launch coverage rather than a filing.

**The mission statement**, from the company's own site: it is creating "a superlearner that
discovers all knowledge from its own experience, from elementary motor skills through to
profound intellectual breakthroughs", which will "rediscover and transcend the greatest
inventions in human history" — language, science, mathematics, technology.
<https://www.ineffable.ai/> **[solid]** as a quotation of their copy.

**The name.** The site lists seven stated beliefs. One of them is *Ineffable*: the
knowledge such a system acquires "will be too profound to be described by human language."
**[solid]**, and it is the single most useful sentence on the slide — the whole callback
turns on it.

**Reported but worth a hedge.** Silver has been quoted saying a success would be "a
scientific breakthrough of comparable magnitude to Darwin", that he considers this his
life's work, and that money he makes from it will go to high-impact charities.
**[reported]** — press interviews at launch. Use "he has said" and move on; the lecture
does not need them.

**The thing to say out loud so the slide isn't an advert.** No product, no paper, no public
result. The bet is a thesis plus a track record. The known weakness is the one slide 9g
already gave them, and the paper itself concedes it: the method has never made the jump
from closed problems with an exact reward to open problems without one. That is the whole
question, and it is unresolved.

## Questions students actually ask

**"Isn't the reward just a label? How is this different from supervised learning?"**
A label tells you the right answer for *this input*. A reward tells you how well a whole
sequence of decisions turned out. You can't do gradient descent on "4 out of 10" directly
— there is no target to subtract from. That's why the machinery is different.

**"Couldn't you just try every possible sequence of moves?"** In Pong, 2 actions × ~1,000
frames is 2^1000 sequences. In Go, ~250 legal moves and games ~150 moves long. The
universe doesn't contain enough time. Search must be guided by something learned.

**"If AlphaGo Zero learned with no human data, why did AlphaGo use human data at all?"**
Speed and stability — starting from human play was a shortcut, and part of what Zero
showed is that the shortcut wasn't necessary, and was in fact a ceiling.

**"Does this mean RL will replace supervised learning?"** No. They're being combined.
Pre-train on human data to acquire broad competence, then fine-tune with a reward signal
to shape behaviour. That is how the chatbot they used this morning was built.

**"Where does the reward come from in the real world?"** This is the best version of the
question and the honest answer is: often from a human, or from a model trained to imitate
a human's judgement. Which brings all the specification-gaming problems from slide 11
back, at scale.

**"Is this how humans learn?"** Partly, apparently — the dopamine result is real. But
humans are enormously more sample-efficient, and the main hypothesis for why is that we
bring structure: objects, physics, causality, other minds. Closing that gap is one of the
live research problems.

## Further reading to give them

- **Karpathy — Pong from Pixels.** <http://karpathy.github.io/2016/05/31/rl/> Readable
  in an evening, and the code is 130 lines.
- **Silver — RL Course**, lectures 1–2 at minimum.
  <https://www.youtube.com/watch?v=2pWv7GOvuf0>
- **Sutton & Barto — Reinforcement Learning: An Introduction**, 2nd ed., free at
  <http://incompleteideas.net/book/the-book-2nd.html>. Chapters 1 and 6.
- **AlphaGo** (2017 documentary). Ninety minutes, and it is the best possible homework
  after slide 18.
- **Spinning Up in Deep RL** (OpenAI) — <https://spinningup.openai.com> — for the ones
  who want to implement.
- **Silver & Sutton — Welcome to the Era of Experience** (2025). Nine pages, no maths, and
  it is the argument of this lecture pointed forwards.
  <https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf>
- **Is Human Data Enough?** — Google DeepMind: The Podcast, Hannah Fry with David Silver.
  <https://deepmind.google/the-podcast/> The listening homework after slide 33.

---

# Appendix D — Graphics still to be built

The deck carries three dashed placeholder blocks. Each names what goes there.

| Slide | Placeholder | Status |
|---|---|---|
| ~~1c~~ | *(slide cut — see note below)* **Three learning signals, in 3D.** The same weight-space terrain under three states of knowledge: Hebb climbing correlation with no target; supervision on a fully lit error surface; reinforcement on the identical terrain, fogged, lit only where the walker has stepped, with the height reading arriving several steps late. | homeless |
| 9a | **The pixel pipeline, live.** A real Pong frame with crop, downsample and binarise applied under the viewer's control, and the 100,800 → 6,400 collapse shown as it happens. | proposed, not agreed |
| 9d | **Credit assignment, playable.** Play a point; a +1 lands forty frames late; the demo asks which action earned it, then shows what policy gradients actually do to all of them. | proposed, not agreed |
| 9f | **Expected-return landscape, 3D.** The true gradient, invisible, against the noisy sampled estimate the algorithm actually climbs — the RL sequel to Lecture 1's `weight-space-ascent.html`. | proposed, not agreed |

Non-interactive assets still wanted — **sources for all three are now in Appendix E**:

- **The Breakout tunnel clip** — DeepMind's own footage, for slide 16. Either a local file
  in `attachments/` or an embedded link. *Source found; download it rather than stream.*
- **Move 37 board diagram** — an SVG of the position would be better than prose on slide
  5b, and it is a small job. *Draw it from Menick's reconstruction.*
- **Thorndike's learning curve** — escape time against trial number. A ten-line matplotlib
  plot; it makes the "gradual, not insight" point instantly. *Numbers from Chance's paper.*

**7c** should arguably *be* Schultz's three-panel dopamine figure with almost no words on
it; the source is in Appendix E.

**Two orphans from the January cuts.** The three-signals 3D graphic was built for 1c, which
is gone — its content now lives as two lines on slide 5, and the graphic has nowhere to go
unless you want it there. AlphaGo Zero's Elo curve was for 5c, also gone. Neither is lost;
both are simply homeless, and worth deciding about rather than forgetting.


---

# Appendix E — Media and interactive assets

Found for this deck; nothing here is embedded yet. Everything is free to view. Where a
video is long, the useful excerpt is given, because nobody wants ninety minutes of
documentary at 11am.

## Video — the three clips that carry a slide on their own

| Slide | Asset | Use |
|---|---|---|
| 3a / 3b | **DeepMind's own DQN Breakout footage** — <https://www.youtube.com/watch?v=V1eYniJ0Rnk> (DeepMind, 2015) and <https://www.youtube.com/watch?v=eG1Ed8PTJ18> | The tunnel. Play the "after 600 episodes" segment only. This is the Appendix D item marked *wanted*; download rather than stream, the room's wifi will let you down. |
| 5b | **AlphaGo — The Movie**, full documentary, free and official — <https://www.youtube.com/watch?v=WXuK6gekU1Y> | **Move 37 is at roughly 49:00.** Ninety seconds either side is all you need: the commentators calling it a mistake, then the room realising. It does more than any diagram will. Also the obvious homework. |
| 8b / 9g | **OpenAI — Emergent tool use from multi-agent interaction** — <https://openai.com/index/emergent-tool-use/>, paper at <https://arxiv.org/abs/1909.07528> | Hide-and-seek. Six strategies and counter-strategies emerge from nothing but competition and a reward. The best thirty seconds of evidence that exploration invents things, and it gets a laugh when the seekers learn to surf on a box. |

Second-tier video, if a slide needs filling:

- **OpenAI — Solving Rubik's Cube with a robot hand** — <https://openai.com/index/solving-rubiks-cube/>.
  Pairs exactly with slide 9g's cup example, and the stuffed-giraffe prod is the memorable bit.
- **DeepMind — bipedal robot football from deep RL** — project page with videos at
  <https://sites.google.com/view/op3-soccer>, paper <https://arxiv.org/abs/2304.13653>.
  Cheap, small, visibly *learned* rather than choreographed — which is the point.
- **Silver's own RL course**, lecture 1 — <https://www.youtube.com/watch?v=2pWv7GOvuf0> —
  already linked on slide 11.

## Interactive — things they can open in a browser during the lecture

- **Karpathy's REINFORCEjs** — <https://cs.stanford.edu/people/karpathy/reinforcejs/>.
  Four demos, all in-browser, all with a visible policy:
  - *GridWorld: DP* — policy iteration, arrows updating on a grid. The clearest picture of
    Bellman's recursion there is, and it belongs next to **slide 12**.
  - *GridWorld: TD* — SARSA and Q-learning learning the same grid from experience, no
    model. Put the two gridworlds side by side and the model-free point makes itself, at
    **slide 13**, where δ is now defined.
  - *PuckWorld* and *WaterWorld: DQN* — a neural policy in continuous state, learning live.
    Good ambient background while you talk through **slide 15**, since exploration is
    visible in the flailing.
  This is the single highest-value link in this appendix: it is by the author of the Pong
  post they are about to read, it needs no install, and it runs on a phone.
- **Multi-armed bandit simulators** for slide 15, if you want ε-greedy shown rather than
  described: <https://github.com/mweglowski/bandit_problem_simulator> (visual, ε-greedy
  front and centre) and the Neuromatch Academy bandit tutorial, which runs in Colab —
  <https://compneuro.neuromatch.io/tutorials/W3D4_ReinforcementLearning/student/W3D4_Tutorial2.html>.
- **GriddlyJS** — <https://arxiv.org/abs/2207.06105> — web IDE that embeds trained policies
  as TF.js components. Only worth it if you decide to build the Appendix D graphics on top
  of something rather than from scratch.

## Diagrams and figures to lift

- **Schultz's dopamine figure — the three panels.** Unpredicted reward: a burst. Predicted
  reward: the burst has migrated onto the cue and the reward itself produces nothing.
  Predicted reward omitted: firing drops *below* baseline at exactly the expected moment.
  Clean open-access reproductions in Niv & Schoenbaum's PNAS piece —
  <https://www.pnas.org/doi/10.1073/pnas.1014269108> — and in Schultz's own open review,
  <https://www.tandfonline.com/doi/full/10.31887/DCNS.2016.18.1/wschultz>. **Slide 7c
  should be this figure and almost no words.** The third panel is the argument.
- **AlphaGo Zero's Elo curve**, Nature 2017, figure 3 — the from-scratch line crossing
  AlphaGo Lee at three days and running on to ~5,200 at forty. Open PDF of the paper at
  <https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf>; DeepMind's
  blog version is friendlier — <https://deepmind.google/blog/alphago-zero-starting-from-scratch/>.
  Slide 5c has since been cut, so this now has no home unless the Zero story comes back.
- **Move 37 board position.** DeepMind's AlphaGo page has stills —
  <https://deepmind.google/research/alphago/> — and the clearest written reconstruction,
  with diagrams and an argument about *why* the move is good, is John Menick's essay,
  <https://www.johnmenick.com/writing/move-37-alpha-go-deep-mind.html>. Still worth drawing
  the SVG ourselves (Appendix D), but Menick is the reference to draw it from.
- **Thorndike's puzzle box and learning curve.** Chance's historical paper reproduces the
  original apparatus and curves and is a free PDF —
  <https://www.appstate.edu/~steelekm/classes/psy5300/Documents/chance-72-433.pdf>. If you
  would rather generate the plot, Appendix D already lists it as a ten-line matplotlib job;
  Chance is where the numbers come from.

## Still to make ourselves

Nothing above replaces the four interactive graphics in Appendix D. What it does do is
cover the three *non-interactive* assets that appendix listed as wanted: the Breakout clip
(DeepMind's own footage, above), the move 37 diagram (draw from Menick) and the Thorndike
curve (numbers from Chance).
