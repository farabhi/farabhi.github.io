---
layout: post
title: "Same code, same seed, different answer"
date: 2026-09-03
tags: [debugging, reproducibility, numpy, sql]
description: "A defect that never produced a wrong answer - only a different one each time I asked. And how that kept a dead result alive for months."
---

The same code, with the same fixed seed and the same input file, gave different answers on different machines. I noticed that early, but waved it away. The differences looked cosmetic, less successful runs didn't disprove my findings, so I lulled myself with a comfortable story about a stochastic process.

That ended when I tried to increase the sample size. Unexpectedly, the result I had been building on collapsed, and with it my reason for not caring. The findings were not supported by my experiment anymore, so I naturally suspected the machine-related variances were early warning signs.

The two problems turned out to be unrelated in cause, but connected in how one obscured another. The machine-dependent noise made the early signs of real failure look like more of the same. Had I fixed the architecture problem when I first saw it, the genuine one would have stood out sooner.

## The setup, stripped to its bones

The pipeline starts by selecting a "middle band" of samples from a dataset - discard the extremes, keep the moderate ones. The code did what looks entirely reasonable:

```python
df_sorted = df.sort_values('value')
middle = df_sorted.iloc[1500:4500]   # drop the tails, keep the middle
```

Nothing special, right? Sort rows by a column, select rows by position. Anyone doing data processing had done it many times. Yet this is exactly where my experiment was broken. 

## Ties plus a positional slice

The first thing to mention is that the `value` column is full of ties. Out of 6,000 rows, about 5,400 shared their value with at least one other row. The measurement only has so much precision, so the same number shows up again and again.

Sorting by a column with ties gives you a *partial* order, not a *total* one. The sort will correctly put all rows with 7.2 value in between rows with 7.1 and 7.3. But internal arrangements of rows with 7.2 is never guaranteed. And this is harmless if you don't care about the order *inside* a tie group. Now, if you pair sorting with `iloc[1500:4500]` you might start caring - I know I should have. If the positions 1500 and 4500 land in the *middle* of tie groups, then sorting and slicing might deliver different samples each time you perform them. 

## The part that made it hardware-dependent

Yet it didn't. The sample was reproducible on a single machine, no matter how many time you run it. It's only when I changed the host I'd get variation and potential complete inversion of experiment results. My code didn't use accelerators or CUDA libraries, so why different hosts would have different outcomes?

The answer is it's because the thing breaking the ties is NumPy's sort implementation, and NumPy selects its sort routine based on the CPU it is running on. Some processors
support a vectorised instruction set called AVX-512, which lets NumPy sort through a code path that processes several values at once. Processors without it take another path. Both are correct - each produces a properly sorted array -but the value ties are resolved differently. No arithmetic changed. Not one number was computed differently. The *ordering of equal values* changed, and a positional slice turned that invisible difference into a different sample -  which moved the apparent success rate of the whole experiment from 8/15 to 14/15 on identical input data.

One possible way to confirm it is by running the identical pipeline across different cloud runtimes - I executed them on 6 separate ones. They split into exactly two groups with two different results, and the split followed one thing: whether the CPU reported AVX-512 support. Furthermore, it's impossible to know beforehand which hardware you land on. Neither runtime type, nor vendor would tell you in advance if AVX-512 will be present or not. I had to keep the diagnostic fingerprint in my experiment script always on to know if the sample will be reproduced or not.

## What the defect did, and what it didn't

Let me be precise - the sort defect did not cause my result to crumble. Had the method genuinely worked, my imperfect sort would not have made it fail - it would have produced *real but variable* outcomes, differing by host machine. At worst, I'd have had a numerical reproducibility problem and nothing more.

This variability muddied the water considerably and obscured the real problem. Every architecture gave a plausible answer. Different runs gave different plausible answers. Variability in success and hits seemed to be intrinsic to the experiment and consistent with its principal claim.

That's why the chain *more data → collapse → traced to the sort bug* was my first instinct and was completely wrong. Instead, two separate facts about the same project need to be stated clearly:

1. The method didn't work. That was always true; a larger sample made it visible.
2. The pipeline was non-deterministic across architectures. That was also always true, and it is what stopped me from establishing (1) months earlier.

They share connection even though they don't share the cause. The collapse is what finally made me take the anomaly seriously. The anomaly is why the collapse took so long
to be admitted and addressed.

## The fix, and the thing the fix is an instance of

There are several ways to fix this. I chose to make the sort key total, by adding a tiebreaker that is unique per row:

```python
df_sorted = df.sort_values(['value', 'id'])   # 'id' breaks every tie
middle = df_sorted.iloc[1500:4500]
```

Now every row has a definite position, no ties remain for any implementation to resolve differently, and every machine produces a byte-identical ordering. After the change, all six runtimes produced identical output down to a hash of the selected sample. Sample reproducibility has since held across a CPU vendor change, an AVX-512 boundary, and both NumPy and Python version bumps.

Again, fixing sample reproducibility didn't save the experiment from being refuted. It just meant that from then on I could distinguish between different sources of variability. Which is a must have for learning anything from an experiment at all.

There is a second fix worth mentioning, because it also works. Many sort functions accept a "stable" option (kind='stable' in NumPy), which preserves the input order of tied elements. Stability is a property of the algorithm, not of the hardware, so a stable sort would have given me the same pool on every machine. Had I reached for it, this post would have a shorter middle section.

I prefer the total key for reasons that are real but not dramatic. Stability makes the output a function of the input order - a property nothing in my code sets, records, or checks. A total key makes it a function of the data: shuffle the dataframe first and you get an identical pool. And the rule is visible. sort_values(['value', 'id']) states in one line how ties are broken; kind='stable' defers the question to however the rows happened to arrive.

Neither protects you from the larger thing. If the underlying dataset changes - a new release, a different query, more rows - the pool changes too, and no sort key saves you. That's a separate problem with a separate answer, which in my case was to commit the exact snapshot the results came from.

But using a total sort key was just a recipe for this particular experiment. There's a wider lesson to be learnt from it that could be applied to future experiments as well, even if recipes will look different. 

**The sort key was a proxy. The rows were the payload.**

If my dataframe had held nothing but `value`, ties would have been genuinely harmless. Any tie ordering produces the same multiset of numbers, and nothing downstream could tell the difference. In that world the slice is safe and this post doesn't exist.

But the rows carried something else - in my case molecular structures, and the downstream code depended heavily on *which structures* it received. `value` was
only a convenient handle for choosing among them.

So when I sorted on `value` alone, it was as if I made a claim in my python code: *rows with equal `value` are interchangeable for my purposes.* That claim is true of the numbers. It is false of the molecules. And nothing in the code marked the difference, because the proxy and the payload travel in the same object and get selected by the same
operation.

That's the general shape, and it isn't specific to sorting:

> Know which column you actually care about, and don't let a different column
> stand in for it silently. If a proxy has to do the selecting, be explicit
> about what the proxy's equivalences mean for the payload.

The total-key fix works precisely because it retires the claim. With a unique tiebreaker there are no equivalences left to be wrong about.

## You have probably seen this bug before

If it feels familiar, it's the classic SQL pagination bug wearing different
clothes:

```sql
SELECT * FROM items ORDER BY created_at LIMIT 20 OFFSET 40;
```

If `created_at` isn't unique - and timestamps collide more often than people expect - then "page 3" isn't well defined. The same row can appear on two pages
or none.

And it fails for exactly the reason above. Nobody paginates in order to collect timestamps. You want the *rows*; `created_at` is the proxy you're ordering them by. Equal timestamps do not mean interchangeable rows, but `ORDER BY created_at` asserts that they do. The fix is the same: add a unique tiebreaker, and the false claim disappears. SQL doesn't offer the stable-sort shortcut - tables are unordered by definition, so there's no input order to preserve. Either you supply a total key, or which rows land on which page is up to the planner.

So, the practical rule and the one underneath it:

> Any time you select data by position - a slice, a limit, an offset, "the top N" - the sort that defines those positions must be total.

And more generally: 

> When you order by one thing to select another, you are asserting that the first thing's ties are irrelevant to the second.

And one about the debugging, which cost me the most:

> When two things are wrong at once, the noisy one hides the quiet one. My sort defect produced results that varied architecture to architecture, and that variation was exactly the  shape of the evidence I needed to see that the method itself wasn't working. I spent months reading one problem's symptoms as the other's noise.

---

*This came out of a reinforcement-learning experiment on molecular lead optimization; the code, data and full write-up are at [github.com/farabhi/rl-lead-optimization](https://github.com/farabhi/rl-lead-optimization). Fixing this made the dataset reproducible across machines. It did not make the experiment reproducible - that's a separate problem, and a stranger one, which I'll write about next.*
