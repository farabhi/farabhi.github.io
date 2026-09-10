---
layout: post
title: "Taking apart a result that (never) worked"
date: 2026-09-10
tags: [reinforcement-learning, reproducibility, machinelearning, research]
description: "My experiment confidently proved the expected result. Pulling on the one loose thread cost me the result and left me with something better."
---

Sometimes, being restless about small things is a good trait. It helps to dismantle a beautiful facade that hides broken machinery inside, and as such clears the ground for building something that does actual work. That's exactly what happened with my RL experiment, and this is my story about it.

## The thing that worked

The setup: a known EGFR inhibitor (a molecule that blocks EGFR, a receptor whose overactivity drives several cancers), fifteen steps to improve it, a composite score combining predicted activity with drug-likeness properties. Five policies - Random, ε-greedy, triggered ε-greedy, UCB, Thompson Sampling - differing in how they trade off ***exploiting*** what they've learned / ***exploring*** what they haven't. Tabular Q-learning, linear value function, three random seeds.

The results were clean. Learned policies crossed the success threshold more often than random modification. Some cleared conventional significance. The weight matrices showed distinct, interpretable functional-group preferences - the policies had clearly learned *something*, and it looked like the something was useful.

## Bothersome snags

However, there were some ***irregularities*** that showed up quite early.  The scene was set when I decided to try different hosts just to see if there would be any speedup, as the experiment took a considerable time to run. I didn't expect much as I didn't use any CUDA-like code, so GPUs and TPUs shouldn't have had much to chew on. It transpired exactly as expected - while there was a speed increase, it was moderate and was explained by faster CPUs on those hosts. What I didn't expect was that the performance of RL policies would differ between hosts. While RL policies still outperformed baseline and random modifications, they showed different numbers of hits and different statistical significance, and as a result, different rankings.  What kept gnawing at me in the back of my mind was the fact that I couldn't pinpoint where the differences came from. Changing machines might change some arithmetics, but it shouldn't have affected the rankings and numbers of hits. 

Why it stayed just an 'irritation' and didn't lead to the project overhaul immediately was the fact that its results essentially stayed the same. Difference in numbers didn't look like much when the significance boundary was crossed confidently. My thought process at that time was that I just had a stochastic process. It didn't challenge results as it still had a stable conclusion underneath. 

Ironically, I used the same reasoning to wave away another concern - I had doubts that single matrix could be a good retaining mechanism for the policies in my experiment. The initial molecules that policies are called to improve on are chemically different compounds, they would require different modifications to increase their useful traits. Single matrix retaining preferences for all possible groups of initial scaffolds seemed like a naive implementation. "Yet it worked" - I kept telling myself, "The stats don't lie".

However, they might hint at something. For example, they might hint that the great results I was witnessing were nothing more than random success. Irritating differences among hosts were exactly that - early signs that something was seriously wrong. Tracing those differences back to their roots and learning invaluable lessons along the way turned out to be the most important part of the project, at least this iteration of it. 

## Pulling the thread

Where the whole fabric started to disintegrate was where it should have been the project's ultimate triumph - increasing the number of experiments (or 'seeds', as each experiment was controlled by a randomizer's seed for reproducibility purposes). As I've mentioned before, the script was quite slow, so I kept the number of experiment runs to a minimum I was comfortable with - just three of them for each policy. When I thought I was ready to share my results with a bigger world, I increased the number of seeds to ten per policy. 

It did turn out to be a better sampling, but a better sampling that showed exactly what my initial results were - merely, the lucky chance. With ten seeds the performance of RL policies degraded dramatically, if not catastrophically. Many of them began to trail behind Random, to the point where it might be just one policy showing a significant improvement over random modifications. 

The catastrophe was not that RL didn't work - in any case, I already had doubts that my setup was enough for RL agents to learn properly. The real issue was something else - namely, unstable results that in different runs would give you every answer possible, however contradictory. I didn't know what, if anything, worked in my script and what needed to be done to fix what didn't work. 

Adding it all together - single matrix doubts, variability across hosts, collapsing with increased number of experiments - I realized that all 'irregularities' need to be accounted for properly if I wanted the project to survive.

## First patch

Fixing the script started somewhere unexpected: one line of data preparation.

The pipeline selects a middle band of compounds (EGFR molecules from the curated data set) by sorting on measured score and taking a positional slice. But those score values are mostly ties - 5,400 of 6,000 rows share a value with some other row - and a sort with ties leaves the tied rows in an order nobody defined. NumPy picks its sort routine based on the CPU it's running on. Machines with the AVX-512 instruction set ordered the ties one way; machines without it ordered them another. Different molecules entered the pool, and everything downstream moved with them.

No arithmetic differed. Not one number was computed differently. The ordering of equal values changed, and a positional slice turned that into a different dataset. I've written that part up [separately](/2026/09/03/same-code-same-seed-different-answer.html).

Fixing it was one extra column in the sort key. The dataset became reproducible across every environment I've tried since: a CPU vendor change, an AVX-512 boundary, NumPy and Python version bumps.

That should have been the end of it. Instead it just started a chain reaction.

## The patch won't stop untangling

Same data on every machine, same seeds - and the policies still diverged. Now my concern reached the point where I put everything else aside and concentrated on tracing it back to its exact sources. 

After several debugging experiments I realized that variations come from last-bit differences in floating point results that might differ between machines.

The mechanism is simple to explain. A policy computes action values by multiplying a state vector by a weight matrix, then applies the argmax. That arithmetic is deterministic on any given machine - repeat a run and you get identical bits - but not across machines. Different processors execute it along different instruction paths and can land on a different final bit. If two action values differ only there, argmax returns a different action on one machine than on another. That action changes the molecule, which changes the reward, which updates the weights. From that point the two runs aren't just 'a bit different'. They are completely separate and might lead to different outcomes. 

My first instinct was to treat this as a numerical problem and go looking for a sturdier selection rule. That instinct was wrong, and understanding why is the moment the whole project turned over.

A one-bit gap between two action values means the policy considers those actions *equally good*. Choosing between them arbitrarily is just an exploration phasein RL policy, there's nothing wrong with that. It doesn't matter whether exploration is triggered by a quadrillionth value or by a tenth after rounding it up. What actually is wrong is that a policy keeps deciding at the last bit throughout the whole experiment. 

**If a trained policy is still deciding at the last bit, it means it never separated the signal from the noise**. A policy that had learned something would have gaps between best and second-best action many orders of magnitude larger than machine precision, and a rounding difference couldn't reach them. The sensitivity I'd been treating as a numerical nuisance was a measurement, and what it measured was that nothing had been learned.

It was quite a harsh landing, so naturally I asked myself how it was possible to be in the dark about it for so long.

## What was actually stitching it all together

The environment. Specifically, a rule I'd written without thinking hard about it.

Each episode takes a fifteen-step walk and reports the *best molecule seen anywhere along the way*. That's Monte Carlo maximisation: draw many samples, keep the maximum, and the maximum improves with the number of draws. It's the retention that produces the good results, not the search.

You can see it in what beats what. At the extreme - each method's best fifty molecules - every method clears the screening baseline decisively, Random included. Across everything the methods retained, Random and the screening baseline are statistically indistinguishable, and the learned policies sit slightly below both. The environment's tail is good. Its typical output is ordinary.

Random comes out ahead of every learned policy, which is the detail I find most instructive. Same walks, same retention rule, same budget - and the policy that imposes no preference gets the best tail. A systematic preference that can't tell contexts apart costs you sampling variety and buys nothing back. Consistency without correctness is strictly worse than no consistency at all when you're keeping the best of many samples.

## Why there was nothing to learn

My initial doubt was a key to it all. A single linear weight matrix maps molecular fingerprints to action values across every structural family in the pool at once. When different scaffolds reward different edits - hydroxyl here, methyl there - one matrix cannot hold both, so instead it averages them.

That isn't a failure to detect a signal, rather the model had nowhere to put one. You can see it in the learned weights: every action value ends up negative, and the policy is choosing the least-bad option rather than a good one. And it's the same fact the argmax margins were reporting from another direction. Values that can't separate contexts can't separate actions either.

The policy also controls less than it appears to. It picks which functional group to attach. It doesn't pick the starting molecule - random draw. It doesn't pick where the group lands - also random. The chemical context in which its one decision gets made is entirely outside its control.

## What I'd take from it

I ended with no result and a much better understanding of what a result would require: a total sort key so the data is stable, host discipline so trajectories are comparable, enough seeds that a verdict isn't a coin flip, and a representation rich enough for the policy's one decision to mean anything.

That's a worse outcome than "Thompson Sampling wins" and a more useful one. The winning version would have been a leaderboard entry, and - given how the ranking moved under every perturbation I eventually tried - an unreproducible one.

The lesson I'd actually hand to myself and to whoever'd listen:

> Every inconsistency in your results needs an account, not a label. "It's
> stochastic" is a label. An account names what varies, says why, and predicts
> the variation well enough that you can reproduce it on demand. If you can't
> reproduce your own noise, you don't understand it - and until you understand
> it, you cannot tell it apart from the thing you're trying to measure.

The harder part isn't just knowing that. It's that nothing external will make you do something about it. Nothing was challenging my results. They were good, consistent, and the one loose thread had a plausible-sounding explanation attached. The only thing that made me pull on it was that the explanation never quite satisfied me, and I've come to think that low-grade dissatisfaction is worth more than it feels like at the time. A result nothing is questioning is exactly the one that has to be questioned by you.

---

*Code, data, and full write-up:
[github.com/farabhi/rl-lead-optimization](https://github.com/farabhi/rl-lead-optimization)*
