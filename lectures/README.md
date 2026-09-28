# Lectures

Longer-form material that sits alongside the worksheets: the talk this repo came out of, plus
notes on ideas the worksheets only touch.

Worksheets teach the mechanism you need to run the code. This is for the surrounding argument:
where a technique came from, why it works, what it connects to.

## Lecture 2: learning without a teacher

`02-learning-without-a-teacher/` holds the RL lecture. Numbered 02 because lecture 1, on the
history of neural networks, is still in the vault.

| File | What it is |
|---|---|
| `delivery.md` | The script, slide by slide, with delivery notes and a background section |
| `plan.md` | The plan it was built from |
| `lecture.html` | The rendered deck |
| `attachments/` | Figures the deck uses |

Slides 19 to 27 cover the same ground as worksheets 1 to 3, deliberately. A lecture is spoken
and a worksheet is practised, and meeting an idea twice in two modes is the point.

Four passages in `delivery.md` say things the worksheets do not, and are worth reading even if
you have done all four worksheets:

- Slide 22, the three answers students give to credit assignment, and what each one is called
  in the literature: discounting, baselines and advantage, temporal difference.
- Slide 24, on flattening: turning 80 by 80 into a 6400-vector destroys the fact that adjacent
  pixels are adjacent, and the network has to rediscover it. A convolutional net would not.
- Slide 25, on why REINFORCE converges at all. An action unrelated to the outcome appears in
  good and bad episodes at the same rate, so it gets pushed up as often as down. Nobody
  assigns the credit; the statistics do, slowly. This is the answer to worksheet 3's fifth
  question, which the worksheet leaves open.
- Slide 26, on where the gradient comes from. Supervised: the multiplier is 1 and arrives
  instantly, because a person wrote the answer down. Policy gradient: it is the return, it may
  be negative, and it arrives eighty frames later. Sample inefficiency is not a bug in the
  implementation, it is the price of not having a teacher.

Figures borrowed from Karpathy's post are not redistributed here; the deck links to the
original in their place.

## Notes

| File | About |
|---|---|
| `information.md` | Representation design: transforming raw observations so the task-relevant structure is easier to see. Frame differencing, frame stacking, invariance, sufficient statistics, and why Pong's preprocessing is an instance of something general. |

Interactive demos live in `demos/` at the repo root, because the worksheets use them too.
