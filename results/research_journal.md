- training on full cifar10
- test set of 1000 - using val loader for eval pretext metrics - I shall look out for performance on test set for inpainting
- added status.init for each batch as well as inference
- batch size increased from 100 to 200 - when we increase batch size in conjunction with status.init setting we have more samples to learn and adjust the scrambled starting point
- 25 percent random masking
- inference clamp to 1.0 - why? lower this, as this will not let time for the higher layers to converge on the more abstract features, and lower layers will already settle on the noisy pixels
- I have learning rates with and without status.init - the learning rates I have lowered quite a bit to 0.0001 for w and 0.005 for h. - one area of interest is for lifelong learning during inference we should update both w and h, but this should be gradually lower so that the network does not forget. or maybe there should be sensitive periods for certain layers - e.g. the closer to sensory layers could have a higher rate initially while negligible after some time, while the higher layers should be more plastic. once we learn very basic features from images, there is little need to change those features.
- Increased inference steps from 12 to 24 - have to experiment with the right amount of inference steps. - not sure what works best - too high can even be detrimental - maybe in conjunction with inference clamp alpha this should be tuned
- ok so what am I looking for with regards to reconstruction? if the model has learned good features it shall reconstruct unseen images? well, the hypothesis is that somewhere during reconstruction if we set alpha to lower than 1 and if noise is less than 0.5 and it gives enough signal as to what the image might represent, then the abstract concepts, like car, cat, shall let the model to reconstruct the image to some extent. but the model is not really trained for this. it is just trained to “observe” images. and that is what it does during test. it observes new images that it has never seen. if there is more signal to suggest an image is a cat it will be primed to generate such a rendering - but eventually the noise it will render as well - why wouldn't it?
- on the latest test it seems that we are not learning anything generalisable. this is with w updates off and inference alpha to 0.25. why are we incapable of reconstructing? why is it not easy purely based on h updates to get to accurate reconstructions? why do we need the w updates? if we are allowed to update h during inference why not w? we are fine tuning on the test set in both cases isnt it? but PC is not workable without these updates.
- and so it seems that only lowest layers are active still. there is very little activity and gradient on the higher layers. they are not actively communicating. EXPERIMENT: with random settings with high energy and endogenous activity before attempting sensory data.
- one observation is that the model has no issue with reconstructing images using only the last layer over 512 steps. we get perfect reconstructions. what we have trouble with is using the higher level abstractions to generate the images in high fidelity. I have tested with enabling disabling forward flag.
- so after some time the closest vode already perfectly reconstructs  the image. if I run activation from that. we would get the perfect image. but if we pass the activation from the higher level learned latent, we get a more fuzzy image. this makes sense actually. with 1 blocks and with weight updates I get perfect reconstruction as well. but if I turn off weight updates then I get only very faint resemblance or blacks out completely in later epochs. possibly overfits the weights and it becomes impossible to find a set of h values to get reconstructions.
- EXPERIMENT: we may need at least as many hidden steps as we have layers so that h changes can propagate upwards before we commence weight changes. otherwise we will already tweak the sensory layer plus one level up to minimise energy completely, and so  we starve the higher layers of error signal.
- First finish experiment with blocks 3 to search for best LRs. If it shows we cannot go under 0.10 MSE then do the above experiment. => This is indeed the case - as long as h and w updates are simultaneously done, the higher layers are starved of error signal.
- above experiment with `update_weights_every_inference_step`=False works great, with num blocks = 1
  ```
  peak_lr_weights: float = 0.001
  peak_lr_hidden: float = 0.01
  ```
  we approach perfect image generation 0. 009 train mse over 100 epochs: [https://wandb.ai/neural-machines/debug-transformer-search/runs/koea7hk5?nw=nwusergradientracer](https://wandb.ai/neural-machines/debug-transformer-search/runs/koea7hk5?nw=nwusergradientracer "smartCard-inline")
- Did additional grid search to explore lower w updates and higher h updates. It gets best with the widest gap explored:
  ```
  peak_lr_weights: float = 0.0001
  peak_lr_hidden: float = 0.1
  ```
  We get in 10 epochs 0.008 train mse: [https://wandb.ai/neural-machines/predictive_coding_lr_search_nb1/runs/aj0qtjsg?nw=nwusergradientracer](https://wandb.ai/neural-machines/predictive_coding_lr_search_nb1/runs/aj0qtjsg?nw=nwusergradientracer "smartCard-inline")
  And the gradient norms get almost perfectly balanced across vodes.
  Actually we get even lower to 0.001 mse with:
  ```
  peak_lr_weights: float = 0.001
  peak_lr_hidden: float = 0.1
  ```
  [https://wandb.ai/neural-machines/predictive_coding_lr_search_nb1/runs/1hfrzkoz?nw=nwusergradientracer](https://wandb.ai/neural-machines/predictive_coding_lr_search_nb1/runs/1hfrzkoz?nw=nwusergradientracer "smartCard-inline")
  and over 25 epochs 0.0007 mse [https://wandb.ai/neural-machines/predictive_coding_lr_search_nb1/runs/42ruu4dn?nw=nwusergradientracer](https://wandb.ai/neural-machines/predictive_coding_lr_search_nb1/runs/42ruu4dn?nw=nwusergradientracer "smartCard-inline")
- Explore even wider gap? We already get perfect reconstruction. Lets run 100 epoch with best settings first to check if learning is smooth across training.
- continue experiment with num block 2.
- EXPERIMENT: If issues with stability do T_h LR scheduling.
- EXPERIMENT: Adaptive T_h: h steps to break if below certain energy - maybe per layer separately
- EXPERIMENT: if the energy is large on sensory we shall have very large lr\_h e.g. 0.1 and very small lr\_w e.g. 0.0001 so that the error is propagated properly to the higher layers.
- EXPERIMENT:
  **Gating Mechanisms**: Incorporating mechanisms that can dynamically modulate the flow of prediction errors or states between layers. This is akin to attention but could be simpler forms of gating.
  **Strategic Noise Injection**: Consider if injecting small amounts of noise into hidden states or predictions _during_ the `T_h` inference steps could prevent premature convergence of lower layers and keep error signals flowing.
- EXPERIMENT: can we add something to the learning mechanism that punishes if gradients and energies are drastically different across layers? is that possible to factor into the loss function? like in the brain what process could signal that the high energy needs to be balanced but in a way that is not local but global, and so requiring to propagate the energy across the network? how could this be done?
- Applied exponential LR scaling, so we can scale up progressively the gradients for higher layers, as they were drying up. Now I get 0.004 mse with 4 blocks: [https://wandb.ai/neural-machines/debug-transformer-search/runs/4d6l5hin?nw=nwusergradientracer](https://wandb.ai/neural-machines/debug-transformer-search/runs/4d6l5hin?nw=nwusergradientracer "smartCard-inline")
- moving onto num blocks 5 with same settings blew up. I had to apply gradient clipping, but training was unstable: [https://wandb.ai/neural-machines/debug-transformer-search/runs/hcj6iu6c?nw=nwusergradientracer](https://wandb.ai/neural-machines/debug-transformer-search/runs/hcj6iu6c?nw=nwusergradientracer "smartCard-inline")
- gradient clipping seems to mess up training somehow, if I set to 100, 1000, 10000 - they were not working well - it needs further experiments to see how to use it properly
- first let’s figure how to use grad clipping properly with current best 4 block setup: running hyperparam search on grad clipping: [https://wandb.ai/neural-machines/pc-search-elegant-atlas-debug_tiny_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-search-elegant-atlas-debug_tiny_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses "smartCard-inline") - this showed great result with grad clip of 1000 - it got to 0.006 mse in only 10 epochs, compared to 0.004 in 50 epochs without clipping
- Now running experiment again with 5 blocks: `lr_h=0.1`, `scale_base=1.3`, `clip_norm=1000.0`- it is unstable, blew up after 17 epochs; [https://wandb.ai/neural-machines/debug-transformer-search/runs/lauvjvkf?nw=nwusergradientracer](https://wandb.ai/neural-machines/debug-transformer-search/runs/lauvjvkf?nw=nwusergradientracer "smartCard-inline")
- Decided to try separate h and w grad clipping, running grid search on just clipping h with 5 blocks:  [https://wandb.ai/neural-machines/pc-search-radiant-nexus-debug\_tiny\_nb5/workspace?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain\_mse\_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-search-radiant-nexus-debug_tiny_nb5/workspace?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses "‌")
- **Analysis of 5-Block Search Results:**
  - **Fixed Parameters for this Search:**
  - `num_blocks`: 5
  - `peak_lr_hidden`: 0.1
  - `inference_lr_scale_base`: 1.3
  - `peak_lr_weights`: 0.001
  - `epochs`: 10
  - **Best MSE:** `0.01347585`
  - **Best Params for this 5-Block Search:**
  - `h_grad_clip_norm`: 1000.0
  - `w_grad_clip_norm`: 500.0
- **Interpretation:**
  - The 5-block model is proving harder to train to the same low MSE as the 4-block model, even with differential clipping.
  - `lr_h=0.1` combined with `scale_base=1.3` appears to be a viable, albeit high, learning rate for hidden states, _provided that_ gradient clipping is used.
  - The optimal clipping strategy is sensitive to the number of blocks. `h_clip=1000, w_clip=500` is best for 5 blocks here, while `h_clip=1000, w_clip=100` was best for 4 blocks.
- Running a new grid search with lower lr_h=0.07 [https://wandb.ai/neural-machines/pc-search-serene-titan-debug\_tiny\_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain\_mse\_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-search-serene-titan-debug_tiny_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses "‌") :
- **Analysis of the Latest 5-Block Search Results:**
  - **Fixed Parameters for this Search:**
  - `num_blocks`: 5
  - `inference_lr_scale_base`: 1.3
  - `peak_lr_weights`: 0.001
  - `epochs`: 10
  - **Searched Parameters:**
  - `lr_hidden_candidates`: `[0.1, 0.07]`
  - `h_grad_clip_norm_candidates`: `[800.0, 1000.0, 1200.0]`
  - `w_grad_clip_norm_candidates`: `[300.0, 500.0]`
  - **Best MSE in this search:** `0.01991741`
  - **Best Params in this search:**
  - `peak_lr_hidden`: 0.07
  - `h_grad_clip_norm`: 1000.0
  - `w_grad_clip_norm`: 500.0
  - **Key Observations:**
  - `lr_h = 0.07` **Performs Better:** The best run in this set used `lr_h = 0.07`. The top 6 runs all hover around MSE ~0.02 to ~0.025, and the absolute best one has `lr_h=0.07`. Runs with `lr_h = 0.1` in this search generally performed worse, with the best `lr_h=0.1` run achieving an MSE of `0.0222` (`hclip1000_wclip300`). This suggests that for 5 blocks with `scale_base=1.3`, `lr_h=0.1` might still be too aggressive, and `0.07` is a better starting point.
  - **Comparison to Previous 5-Block Best:** The previous best for 5 blocks (from `..._162145.txt`) was `0.0135` with `lr_h=0.1, h_clip=1000, w_clip=500`. The current search did not replicate or beat this. The run with those exact parameters (`lr_h=0.1, h_clip=1000, w_clip=500`) in _this_ search resulted in an MSE of `0.0376` (Run 4), which is significantly worse. This discrepancy is due to run-to run variance. It gets a bit unstable around epoch 10.
  - **Clipping Norms for** `lr_h = 0.07`**:**
  - With `lr_h = 0.07`, `h_clip=1000, w_clip=500` gave the best result (0.0199).
  - `h_clip=1200, w_clip=300` was next (0.0215).
  - `h_clip=800` performed worse for `lr_h=0.07` (0.0234 with `w_clip=500`, 0.0429 with `w_clip=300`).
- I have realised that grad norms are exceedingly pushed over to Vode0, it is really left skewed now by epoch 10 and gets worse from then on with lr\_h=0.1. With lr\_h=0.07 it is better, by epoch 10 we have similar picture as used to with right skew, but if we let it run to epoch 25 we get left skewed and training explodes.
- Next set of experiments is trying to see if using less aggressive lr scaling helps with blocks=5, and the new clipping settings: Analysis [https://wandb.ai/neural-machines/pc-search-radiant-aurora-debug\_tiny\_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain\_mse\_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-search-radiant-aurora-debug_tiny_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses "‌")
- Analysis: best performing is still lr_h 0.1 - crucial finding is that with 1.3 grad scaling the grad norms get extremely shifter resulting in the earlier observed collaps - but with 1.25 and below 1.2, 1.15, it is already good over 15 epochs at least.
- Now running a longer experiment with these settings to 50 epochs: [https://wandb.ai/neural-machines/pc-search-gentle-voyager-debug\_tiny\_nb5?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-gentle-voyager-debug_tiny_nb5?nw=nwusergradientracer "‌"): with 1.2 scaling I can get all the way down to 0.005 - but training is brittle as verified by subsequent run with multiple seeds: [https://wandb.ai/neural-machines/pc-search-noble-titan-debug\_tiny\_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain\_mse\_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-search-noble-titan-debug_tiny_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses "‌")
- SO, objective is achieved for num blocks 5, but stability could be improved, perhaps further fine tuning, or moving on for now to either blocks=6 or trying out more sophisticated methods that adaptively adjusts the inter layer norm somehow.
- Actually just using LR schedule with warmup\_epochs=0 solves the instability for num\_blocks=5 reaching 0.0056 val MSE and 0.0067 train MSE - can still play a bit around with how many epoch over to decay and what decay rate but so far it is sufficient: [https://wandb.ai/neural-machines/debug-transformer-search/runs/tuajszbs?nw=nwusergradientracer](https://wandb.ai/neural-machines/debug-transformer-search/runs/tuajszbs?nw=nwusergradientracer "smartCard-inline")
- Trying to find optimal params for blocks=6 now. With warmup most settings plateau at 0.1. Without it runs are unstable, and still trying to find optimal lr scale and warmup steps. Oh, yes, moved from warmup epochs to warmup steps: [https://wandb.ai/neural-machines/pc-search-swift-nexus-6block_nb6?nw=nwusergradientracer&panelDisplayName=VodeGradNorm%2Fdistribution_epoch_5&panelSectionName=VodeGradNorm](https://wandb.ai/neural-machines/pc-search-swift-nexus-6block_nb6?nw=nwusergradientracer&panelDisplayName=VodeGradNorm%2Fdistribution_epoch_5&panelSectionName=VodeGradNorm "smartCard-inline")  => done initial grid search. Too much steps, e.g. from 50 throttle learning and got stuck at 0.1. Now narrowing down to warmup steps between 10 and 30. => [https://wandb.ai/neural-machines/pc-search-cosmic-nexus-6block_nb6?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-search-cosmic-nexus-6block_nb6?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses "smartCard-inline")  no breakthrough really, step=20 seems best.
- Now experimenting with the new `use_vode_grad_norm`parameter. when using `vode_grad_norm_target`=1.0 then even the previous successful 5 block setup fails: before: [https://wandb.ai/neural-machines/debug-transformer-search/runs/tuajszbs?nw=nwusergradientracer](https://wandb.ai/neural-machines/debug-transformer-search/runs/tuajszbs?nw=nwusergradientracer "smartCard-inline")   after: [https://wandb.ai/neural-machines/debug-transformer-search/runs/x9uy2qk3?nw=nwusergradientracer](https://wandb.ai/neural-machines/debug-transformer-search/runs/x9uy2qk3?nw=nwusergradientracer "smartCard-inline")
- Extensive search around norm target  and lr_h smaller than 0.1:
  - Run: [https://wandb.ai/neural-machines/pc-search-serene-phoenix-5block_dev_nb5/panel/p0fl3opf1?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-serene-phoenix-5block_dev_nb5/panel/p0fl3opf1?nw=nwusergradientracer "smartCard-inline")
  - Results:
  - we need at least 0.1 lr_h.
  - grad norm target 20 is not good, 50 and 100 are roughly on same level
  - all this is only up to 10 epochs. needs to run longer. But before that:
- Experiment: test larger lr\_h-s: idea = because we no longer scale the norms, the higher norms have smaller effective lr\_h maybe we need higher values than before.
  - Test `lr_hidden_candidates = [0.11, 1.0, 0.5, 0.3, 0.2, 0.1, 2.5]`
  - Test wider range of grad norm targets: `vode_grad_norm_target_candidates = [500, 200, 100, 75, 50]`
  - Run: [https://wandb.ai/neural-machines/pc-search-vibrant-aurora-5block_dev_nb5?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-vibrant-aurora-5block_dev_nb5?nw=nwusergradientracer "smartCard-inline")
  - Results: not good its all over the place
  - Interpretation: just realised that this approach is flawed. The overall errors come down to a fixed high level and stay there. I think what happens is that even if there is no error, because we are forcing the gradients to a potentially high value, we are shifting the activations even if they should not be really shifted, this hiders the optimisation.
- Experiment: going back to stabilising block=5 setup with my simple lr scaling method.
  - Test: `inference_lr_scale_base_candidates = [1.2, 1.25]`
  - Test: `warmup_steps_candidates = [0, 100, 250]`
  - Test: `seed_candidates = [42, 50, 60] `
  - Test: `h_grad_clip_norm_candidates = [750, 1000, 2000, 5000] `=> noticed in an earlier experiment with block=4 500 was too restrictive but 1000 showed advantage over 10,000. at 10,000 it was already equivalent to None as it did not ever kick in. There can be a sweet spot somewhere I haven't yet figured.
  - Run:
  - Results: 750 was bad
- Experiment:
  - `inference_lr_scale_base_candidates = [1.2, 1.22, 1.24]`
  - `warmup_steps_candidates = [0, 100, 250]`
  - `h_grad_clip_norm_candidates = [1000, 2000, 5000] `
  - `seed_candidates = [42, 50] `
  - Run: [https://wandb.ai/neural-machines/pc-search-swift-voyager-5block_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-search-swift-voyager-5block_nb5?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses "smartCard-inline")
  - Results:
    - run was interrupted due to power outage but have lots of results
    - `h_grad_clip_norm_candidates`:
      - `inference_lr_scale_base_candidates`: 1.2
        - previous run confirmed that `h_grad_clip_norm_candidates`=750 is bad;
        - now we see that 1000 is bad too: one seed reached 0.0095 by the very end, but kept hovering around 0.012 for almost throughout; the other seed diverged early;
        - but 2000 is very good - much better, goes lower faster and does not diverge - both to them able to reach about 0.008 by around epoch 50;
        - 5000 goes down even faster than 2000 and reaches lower minimum:
          - 0.0061 with one seed by epoch 38, but other seed only reaches 0.0076 by epoch 47 - but both of them oscillate a bit and reaches their minimum after a relative surge;
      - `inference_lr_scale_base_candidates`: 1.22:
        - with `h_grad_clip_norm_candidates`=1000 it is reaching on average 0.11-0.10 - with one of the seeds accidentally reaching 0.0077 - but then jumping back to average 0.13-0.10 ;
        - with `h_grad_clip_norm_candidates`=2000 one of the seeds reaches slightly lower than than with grad clip norm 1000 but still hovers around 0.10-1.12. the other seed starts out higher then all the others and stays worse;
        - with `h_grad_clip_norm_candidates`=5000: one of the seed starts out higher then the others, but eventually reaches to 0.0084 for 2 epochs but then diverges; the other seed reaches 0.009-0.008 territory and stays there more stably, but in the end slowly diverges by epoch 10;
      - `inference_lr_scale_base_candidates`: 1.24
        - with `h_grad_clip_norm_candidates`=1000 one seed quickly diverges; the other seed reaches about 0.01-0.011 - it reaches for 0.0087 for one epoch after a relative surge => these minima after surge seem interesting, it suggests the model can't escape some local minima, but after some major disruptions it found the way;
        - with `h_grad_clip_norm_candidates`=2000 we only have one seed as training got interrupted - even the single seed we have got interrupted at epoch 62 - but this was the BEST RUN nicely coming down and reaching 0.0052 by epoch 52;
    - `warmup_steps_candidates`: warmup steps do not help - none of the runs reached below 0.01 when we had 100 or 200 warmup steps. => can still experiment with very small values [2,5,10] but not sure how meaningful that will be
  - Interpretation and next steps:
    - have to explore more `h_grad_clip_norm_candidates` from 2000 up; maybe [2000, 3500, 5000, 7500]
    - `inference_lr_scale_base_candidates`: while 1.22 did not perform that well, 1.24 showed great promise; 1.20 still solid; maybe just bad seed luck with 1.24; recommendation [1.20, 1.22, 1.24. 1.26]
    - maybe first small experiment searching the vicinity of the best current candidate of `inference_lr_scale_base_candidates`=1.24
- Experiment:
  - `inference_lr_scale_base_candidates = [1.24]`
  - `warmup_steps_candidates = [0]`
  - `h_grad_clip_norm_candidates = [2000, 5000, 10000]`
  - `seed_candidates = [42, 50] `
  - Run: [https://wandb.ai/neural-machines/pc-search-cosmic-voyager-5block_nb5?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-cosmic-voyager-5block_nb5?nw=nwusergradientracer "smartCard-inline")
  - Results:
    - `h_grad_clip_norm_candidates` = [2000, 5000, 10000]
      - seed 42 diverged for all, but seed 50 was fine for all
      - 5000: this had an edge over 2000 mostly throughout reaching 0.0063 mse vs 0.0067 for 2000
  - Interpretaion and next steps:
    - could explore a bit more around 5000 e.g. [3500, 5000, 6500]
    - divergence is still a concern, need to keep using multiple seeds
    - not sure about `inference_lr_scale_base_candidates` results are inconsistent so far; while previous seed for 1.24 reached 0.0052 the new runs reached max 0.0063-0.0067
    - of more immediate concern than the above is to stabilise training, while helping the model to escape local minima - reaching record lows after short term upswings in mse signals that the model was in minima that it had to escape.
      - a major miss is that momentum has not been tuned for a long time, current value is very low at 0.1; `hidden_momentum_candidates = [0.1, 0.5, 0.95]`
      - I will likely need to re-tune `lr_hidden_candidates = [0.1]`
- Experiment:
  - `hidden_momentum_candidates = [0.5, 0.3, 0.1]`
  - `seed_candidates = [50, 42]`
  - `lr_hidden_candidates = [0.1]`
  - `inference_lr_scale_base_candidates = [1.20]`
  - Run: [https://wandb.ai/neural-machines/pc-search-gentle-atlas-5block_dev_nb5?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-gentle-atlas-5block_dev_nb5?nw=nwusergradientracer "smartCard-inline")
  - Results: 
    - with `hidden_momentum_candidates = [0.5]`: the energy came down abruptly, from 86,000 -> 826 I think it is making lower layers to learn more quicly and starving higher layers. both seeds oscillated widely and diverged
    - with `hidden_momentum_candidates = [0.3]`: loss came down nicely, it is still oscillating a bit, but much less, and nicely converge to around 0.006, with a lowest mse of 0.0054
    - with `hidden_momentum_candidates = [0.3]`: loss came down slower and it stayed around 0.010-0.011 - this contrasts with some earlier runs with same settings that reached lower - it seems that this setting is still sensitive to initial settings, and there is larger run to run variance - also scaling `h_grad_clip_norm_candidates = 5000` to `h_grad_clip_norm_candidates = 2000` could help, but I am not clear on this yet
  - Interpretation and next steps:
    - `hidden_momentum_candidates = 0.3` with `lr_hidden_candidates = [0.1]` seems to be beneficial in faster convergence as well as stabilisign and improving later stages of learning, resulting in consistent mse below 0.008
    - `hidden_momentum_candidates = [0.4, 0.3]` in later stages of training could help escape local minima and lower final mse
    - `lr_hidden_candidates = [0.095, 0.085]` have to try with slightly lower values to pair with the higher momentum
- Experiment:
  - `lr_hidden_candidates = [0.095, 0.085]`
  - `inference_lr_scale_base_candidates = [1.21, 1.23, 1.25]` - adjusted these upwards to get effective lr at highest layer roughly the same as with `lr_hidden_candidates = 0.1`
  - `hidden_momentum_candidates = [0.4, 0.3]`
  - `h_grad_clip_norm_candidates = [5000]`
  - `seed_candidates = [50, 42]`
  - Run: [https://wandb.ai/neural-machines/pc-search-vibrant-zenith-5block_dev_nb5?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-vibrant-zenith-5block_dev_nb5?nw=nwusergradientracer)
  - Results:
    - `lr_hidden_candidates = [0.095, 0.085]`: groupby over lr_hidden_candidates final mse:
      - 0.095: 0.0066
      - 0.085: 0.0099
    - `inference_lr_scale_base_candidates = [1.21, 1.23, 1.25]`: groupby over inference_lr_scale_base_candidates final mse:
      - 1.25: 0.0091
      - 1.23: 0.0081
      - 1.21: 0.0091
    - `hidden_momentum_candidates = [0.4, 0.3]`: groupby over hidden_momentum_candidates final mse:
      - 0.4: 0.0095
      - 0.3: 0.0084
  - Interpretation and next steps:
    - when grouping results together based on best params, it looks like lr_hidden_candidates = 0.095, inference_lr_scale_base_candidates = 1.25, hidden_momentum_candidates = 0.3 performs best.
    - there is still quite large run to run variability with different seeds. would be good to work on stabilising training, before moving on to deeper networks. mostly it reaches below 0.008 but it can often stay flat for long, and have big jumps eventually dropping down
- Experiment:
  - nevertheless, run overnight test testing same settings on num block 6
  - `lr_hidden_candidates = [0.095, 0.085]`
  - `inference_lr_scale_base_candidates = [1.21, 1.23, 1.25]`
  - `hidden_momentum_candidates = [0.4, 0.3]`
  - `h_grad_clip_norm_candidates = [5000]`
  - `seed_candidates = [50, 42]`
  - Run: [https://wandb.ai/neural-machines/pc-search-noble-aurora-6block_nb6?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-noble-aurora-6block_nb6?nw=nwusergradientracer)
  - Results:
    - `lr_hidden_candidates = [0.095, 0.085]`: groupby over lr_hidden_candidates final mse:
      - 0.095: 0.017
      - 0.085: 0.021
    - `inference_lr_scale_base_candidates = [1.21, 1.23, 1.25]`: groupby over inference_lr_scale_base_candidates final mse:
      - 1.25: 0.0136
      - 1.23: 0.0215
      - 1.21: 0.0233
    - `hidden_momentum_candidates = [0.4, 0.3]`: groupby over hidden_momentum_candidates final mse:
      - 0.4: 0.0189
      - 0.3: 0.0253
  - Interpretation and next steps:
    - still very large run to run variability.
    - important: all runs here were early stopped as I changed the LR for validation to match train decayed rates. I need to repeat the run with best settings.
    - hidden_momentum=0.40 seems like a strong candidate for 6 blocks. This is the clearest pattern. 
    - lr_hidden=0.095 still looks promising. 
    - inference_lr_scale_base=1.25 appears to be a good pairing, though 1.21 wasn't far behind with the right momentum/seed.
    - let's try to see if increasing steps helps stabilising the training. use best settings
- Experiment:
  - `lr_hidden_candidates = [0.095]`
  - `inference_lr_scale_base_candidates = [1.25]`
  - `hidden_momentum_candidates = [0.4]`
  - `h_grad_clip_norm_candidates = [5000]`
  - `seed_candidates = [50, 42]`
  - `inference_steps_candidates = [20, 21, 22, 24]`
  - Run: [https://wandb.ai/neural-machines/pc-search-elegant-nexus-6block_nb6](https://wandb.ai/neural-machines/pc-search-elegant-nexus-6block_nb6)
  - Results:
    - able to hit below 0.008 mse but large run to run variance, and oscillatory behavior for all settings
- Experiment:
  - In order to smooth out training, let's experiment again with warmup, as well as lowering h_grad clip.
  - `lr_hidden_candidates = [0.095]`
  - `inference_lr_scale_base_candidates = [1.25]`
  - `hidden_momentum_candidates = [0.4]`
  - `h_grad_clip_norm_candidates = [2000, 5000]`
  - `seed_candidates = [50, 42]`
  - `inference_steps_candidates = [20]`
  - `warmup_steps_candidates = [0, 100]`
  - Runs: [https://wandb.ai/neural-machines/pc-search-gentle-nebula-6block_nb6?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-gentle-nebula-6block_nb6?nw=nwusergradientracer)
  - Results:
    - `warmup_steps_candidates = [0, 100]`: 
    - `h_grad_clip_norm_candidates = [2000, 5000]`: 2000 was clearly superior. Groupby results wihtout warmup runs:
      - 5000: 0.0127 MSE
      - 2000: 0.0098 MSE
  - Interpretation and next steps: 
    - This strongly suggests that for inference_steps=20 and current best LR/momentum settings, a tighter h_grad_clip_norm of 2000 is more beneficial for both achieving low validation MSE and maintaining training stability.
    - Checking lower `h_grad_clip_norm_candidates = [500, 1000, 2000]`
- Experiment:
  - Try to smooth out training with lower `h_grad_clip_norm_candidates`
  - `h_grad_clip_norm_candidates = [500, 1000, 2000]`
  - `lr_hidden_candidates = [0.095]`
  - `inference_lr_scale_base_candidates = [1.25]`
  - `hidden_momentum_candidates = [0.4]`
  - `seed_candidates = [50, 42]`
  - `inference_steps_candidates = [20]`
  - Runs: [https://wandb.ai/neural-machines/pc-search-bold-atlas-6block_nb6?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-search-bold-atlas-6block_nb6?nw=nwusergradientracer)
  - Results:
    - demonstrates a clear trend: for these settings, h_grad_clip_norm=500 is too restrictive, hindering performance. h_grad_clip_norm=1000 is decent, but h_grad_clip_norm=2000 allows the model to achieve significantly better results while maintaining good stability, especially for seed 42 in Run 6 and seed 50 in Run 5.
- Experiment:
  - Try to smooth out training with more steps `h_grad_clip_norm_candidates`
  - `h_grad_clip_norm_candidates = [2000]`
  - `lr_hidden_candidates = [0.095, 0.085, 0.075]`
  - `inference_lr_scale_base_candidates = [1.25]`
  - `hidden_momentum_candidates = [0.4]`
  - `seed_candidates = [50]`
  - `inference_steps_candidates = [30, 40]`
  - Runs: [https://wandb.ai/neural-machines/pc-search-mighty-voyager-6block_nb6](https://wandb.ai/neural-machines/pc-search-mighty-voyager-6block_nb6)
  - Results:
    - Best validation MSE: 0.00246 (with lr_hidden=0.085, inference_lr_scale_base=1.25, inference_steps=40, h_grad_clip_norm=2000, seed=50). This is a new record for and a significant improvement over previous results.
    - Increasing inference steps to 40 (from 20/30) gave a clear boost in performance.
    - Lowering lr_hidden to 0.085 (from 0.095) also helped, especially in combination with more inference steps.
    - h_grad_clip_norm=2000 remains optimal.
    - inference_lr_scale_base=1.25 is still the best scaling factor.
    - Stability: The best run finished all epochs, with final train MSE very close to the best val MSE.
  - Interpretation and next steps: 
    - it is still quite oscillatory, not a smooth downward trend so, not sure whether the overhead of double the steps worth it
    - while further small improvements and stability gains are possible we are now in the regime of diminishing returns
    - The next big leap will likely come from architectural changes or new regularization/normalization ideas
    - Proceed with linear probing or trying to run best settings for longer epochs, or experiment with other LR schedule, and batch sizes
- Experiment:
  - Try 6block config for all block variations to see how general the found params are. It would make it easier to generate and compare linear probing results if I can use same settings.
  - `num_blocks_candidates = [0, 1, 2, 3, 4, 5, 6]`
  - `seed_candidates = [50, 42, 100]`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-boring-hoover?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-boring-hoover?nw=nwusergradientracer)
  - Results:
    - I should have used more ES patience as I reduced validation from 10 to 5 epochs. With block 6 especially it lead to premature ES, but other blocks could have benefitted as well for more patience.
    - All settings reached below 0.008 MSE, so using the same settings seems fine

- Experiment:
  - Try best settings for 10 different seeds to check stability
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-recursing-meitner?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-recursing-meitner?nw=nwusergradientracer)
  - Results:
    - hyperparam_search_results/hyperparam_search_results_6block_20250522_035339.txt
    - All nicely converged
    - Best val MSE 0.0042

- Experiment: 
  - Trying to run best settings for different batch sizes to stabilise training, as well as trying layernorm for vode states
  - `batch_size = [200, 500, 1000]`
  - `use_vode_state_layernorm_candidates = [False, True]`
  - `h_grad_clip_norm_candidates = [2000]`
  - `lr_hidden_candidates = [0.095]`
  - `inference_lr_scale_base_candidates = [1.25]`
  - `hidden_momentum_candidates = [0.4]`
  - `seed_candidates = [50, 42]`
  - `inference_steps_candidates = [20]`
  - Runs: [https://wandb.ai/neural-machines/pc-arch-search-heuristic-perlman?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-interesting-sinoussi?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses)
  - Results:
   - I have stopped the training early with the batch size 200 experiments finished as I did not see benefit of using layernorm.
  - Interpretation and next steps:
    - Just run the experiment on looking at the effect of modifying batch sizes

- Experiment: 
  - Trying to run best settings for different batch sizes to stabilise training.
  - `batch_size = [200, 500, 1000]`
  - `use_vode_state_layernorm_candidates = [False]`
  - `h_grad_clip_norm_candidates = [2000]`
  - `lr_hidden_candidates = [0.095]`
  - `inference_lr_scale_base_candidates = [1.25]`
  - `hidden_momentum_candidates = [0.4]`
  - `seed_candidates = [50, 42]`
  - `inference_steps_candidates = [20]`
  - Runs: [https://wandb.ai/neural-machines/pc-arch-search-interesting-sinoussi?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses](https://wandb.ai/neural-machines/pc-arch-search-interesting-sinoussi?nw=nwusergradientracer&panelDisplayName=Losses%2Ftrain_mse_avg&panelSectionName=Losses)
  - Results:
   - Batch size 200 is much better than the other setings.
   - more importantly there is no visible difference in training stability - still oscillating with higher batch size

- Experiment: 
  - Just out of curiosity testing how robust are the hyperparams when we increase network size, adding more hidden_size and num_heads
  - `hidden_size_candidates = [64, 128, 256]`
  - `num_heads_candidates = [1, 2, 4]`
  - `seed_candidates = [50, 42]`
  - `epochs = [150]`
  - Runs: [https://wandb.ai/neural-machines/pc-arch-search-kind-agnesi?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-kind-agnesi?nw=nwusergradientracer)
  - Results:
   - I have increased epochs to 150 - which hurt the performance of even the original setting, so cannot really draw much conclusion from these runs
  - Interpretation and next steps:
    - Let's stop now and do the linear probing tests



- Experiment: 
  - shuffling 
  - theta 10000 
  - image augmentations from vision_transformer script
  - 500 epochs, decayed to 0.1
  - 3 seeds
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-hardcore-varahamihira?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-hardcore-varahamihira?nw=nwusergradientracer)
  - Results:
    - only finished one of the seeds but it diverged

- Experiment:
  - now running with 75 epochs: 
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-eloquent-jang?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-eloquent-jang?nw=nwusergradientracer)
  - from 3 seeds one seed reached 0.006 with shuffling and augmentations, the others early stopped as modified criteria to train_mse with 10 patience
  - ../results/hyperparam_search_results/hyperparam_search_results_6block_20250523_083624.txt

- Experiment:
  - further change in data augmentation - added cifar10 specific normalisation and simplified augmentations - have to test if converges:
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-friendly-lehmann](https://wandb.ai/neural-machines/pc-arch-search-friendly-lehmann)
  - Results:
    - all stopped, and did not reach much:
    - Achieved Overall Best Validation MSE: 0.015082
    - Run Best Train MSE of Best Val Run: 0.01942340098321438
  - Interpretation and next steps: 
    - it was still downward trend when looking running average. maybe it needs more time with augmentaions to converge

- Experiment:
  - running the same as above but with more patience for 10 seeds:
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-boring-turing](https://wandb.ai/neural-machines/pc-arch-search-boring-turing)
  - Results:
   - all but 1 diverged or got worse from about epoch 35
   - best run 0.013 - none of them reached near 0.008
   - ../results/hyperparam_search_results/hyperparam_search_results_6block_20250523_134529.txt
  - Interpretation and next steps:
    - let's backtrack for now and double down on getting best possible run to do linear probing on


- Experiment:
  - New run to produce model artifact with best settings
  - `seed_candidates = [40, 50, 60, 70, 80, 90, 100]`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-adoring-northcutt?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-adoring-northcutt?nw=nwusergradientracer)
  - Results:
    - Best run: nb6_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd80_epoch69_trainmse0.004531_20250523_173804.npz
    - MSE: 0.004531
  - Linear probing:
    - Vode 0 - Final Test Accuracy: 0.1882

- Experiment:
  - New run with above settings but for 150 epochs and to 0 LR decay
  - `seed_candidates = [80]`
  - `epochs = 150`
  - `lr_schedule_min_lr_factor = 0`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-lucid-maxwell](https://wandb.ai/neural-machines/pc-arch-search-lucid-maxwell)
  - Results:
    - Early stopped as it got stagnating and oscillating wildly

- Experiment:
  - Repeat run with lower weight updates
  - `peak_lr_weights: 0.0001`
  - `seed_candidates = [80]`
  - `epochs = 150`
  - `lr_schedule_min_lr_factor = 0.5`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-sweet-lamport](https://wandb.ai/neural-machines/pc-arch-search-sweet-lamport)
  - Results:
    - Run Best Train MSE of Best Val Run: 0.017332421615719795
  - Interpretation and next steps:
    - I have missed to study the effect of weight LR on stability. This run is very stable with 0.0001 weight LR, but does not reach optimal levels. Also, interesting loss curve, going back up for a while after an initial downward trend, but then keeps coming back down again. Let's first try to add augmentations, to see if it helps the curve be consistently downward. 

- Experiment:
  - Repeat run with augmentations and cifar scaling
  - `use_ssl_augmentations: True`
  - `use_cifar10_norm: True`
  - `peak_lr_weights: 0.0001`
  - `seed_candidates = [80]`
  - `epochs = 150`
  - `lr_schedule_min_lr_factor = 0.5`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-inspiring-wescoff](https://wandb.ai/neural-machines/pc-arch-search-inspiring-wescoff)
  - Results:
    - Did not help with the loss shape initially dropping but coming back up a bit and then dropping again. 
    - Loss comes down much slower than without augmentation, it is harder to fit. But that is expected.
    - It starts oscillating around epoch 45, while mse is only 0.042-0.052
  - Interpretation:
    - Not sure why still oscillates after a while. 

- Experiment:
  - Repeat above but bump up learning rate for weights.
  - `theta = 100`
  - `peak_lr_weights: 0.0005`
  - `use_ssl_augmentations: True`
  - `use_cifar10_norm: True`
  - `seed_candidates = [80]`
  - `epochs = 150`
  - `lr_schedule_min_lr_factor = 0.5`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-angry-williams](https://wandb.ai/neural-machines/pc-arch-search-angry-williams)
  - Results:
    - Too much, oscillates again. Stopped, as it looked that bad.
  - Interpretation:
    - Oscillations clearly get tamed with lower regimes, when we bumped up again they creep back in. But even in the above smooth setting oscillations start to accur around epoch 45. 


- Experiment:
  - Revisiting iPC and resetting h each epoch
  - `use_status_init_in_training: True`
  - `use_status_init_in_unmasking: True`
  - `update_weights_every_inference_step: True`
  - `theta = 10_000`
  - `peak_lr_weights: 0.0001`
  - `use_ssl_augmentations: True`
  - `use_cifar10_norm: True`
  - `seed_candidates = [80]`
  - `epochs = 150`
  - `lr_schedule_min_lr_factor = 0.5`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-busy-chatterjee?nw=nwusergradientracer](https://wandb.ai/neural-machines/pc-arch-search-busy-chatterjee?nw=nwusergradientracer)
  - Results:
    - All diverged and higher than 0.3
  - Interpretation:
    - stop this line of experiments for now - refocus effort on linear probing and what affects linear probing scores

- Experiment:
  - Best 6block settings with data augmentation
  - `use_ssl_augmentations`: with gaussian blur, colorjitter, etc.
  - `theta = 10_000`
  - `seed_candidates = [80, 90, 10, 20, 30, 40, 50, 60, 70]`
  - Run: [https://wandb.ai/neural-machines/pc-arch-search-festive-jones](https://wandb.ai/neural-machines/pc-arch-search-festive-jones)
  - Results:
    - new lowest [so far, still running]: trainmse0.003547:
      - nb6_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd10_epoch75_trainmse0.003547_20250526_155633.npz
      - Linear probe results: 
        Vode Combination: 7, Test Accuracy: 0.2563, Num Features: 64
        Vode Combination: 0, Test Accuracy: 0.1929, Num Features: 48
        Vode Combination: 6, Test Accuracy: 0.1902, Num Features: 64
        Vode Combination: 5, Test Accuracy: 0.1592, Num Features: 64
        Vode Combination: 4, Test Accuracy: 0.1589, Num Features: 64
        Vode Combination: 2, Test Accuracy: 0.1579, Num Features: 64
        Vode Combination: 3, Test Accuracy: 0.1296, Num Features: 64
        Vode Combination: 1, Test Accuracy: 0.1278, Num Features: 64
        - Combination:
          - Vodes _concat_0_1_2_3_4_5_6_7 - Final Test Accuracy: 0.2322
          - Vodes _concat_0_7 - Final Test Accuracy: 0.2984
          - Vodes _concat_0_6_7 - Final Test Accuracy: 0.3051
          - Vodes _concat_0_1_2_3_4_5_6_7 - Final Test Accuracy: 0.2237



Experiment:
  - Creating models for each block variation with standard settings
  - Results: 
    - Blocks 0:
      - epoch10_trainmse: 0.000172
      - Vode 0 - Final Test Accuracy: 0.2692
    - Block 1: 
      - epoch18_trainmse: 0.000436
      - Vode 0 - Final Test Accuracy: 0.2418
      - Experiment with SSL + theta=10_000 + batch_size=500
        - epoch69_trainmse: 0.003560
        - nb1_bs500_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd80_epoch69_trainmse0.003560_20250524_120732.npz
        - had to increase LR to starting 0.095
        - Vode 0 - Final Test Accuracy: 0.1840
    - Block 2: 
      - epoch12_trainmse: 0.002394
      - Vode 0 - Final Test Accuracy: 0.2081
    - Block 3:
      - epoch19_trainmse: 0.002849
      - Vode 0 - Final Test Accuracy: 0.1971
    - Block 6:
      - Vode 0 - Final Test Accuracy: 0.1664



- Try next: 
  - Get back to block1 - and iterate quicly with linear probing as direct feedback on how data augmentation, and regularisation can improve representations; I could also experiment faster with more hidden dim and num heads
  - Block6:
    - oscillation reduction: lower learning rate from 0.0005 ro 0.0003
    - maybe need to revisit iPC
    - have to revisit latent resetting
    - outsource hyperparam search - before that merge the liner probing script to main, and experiment on new branch

Criticism:
  - normal ViT with the same architecture as my 6 block 1 head, 64 dim model reaches 76% accuracy in 64 epochs
  - this means it has the representational capacity for good features. Good enough for this classification.

Ideas to test:
 - validate if latents are extracted correctly for linear probing. when we do 200 batches, we are not averageing the latents right?
 - can we do training like starting from 256 batches, then reduce to 128, 64, 32, 16, 8, 4, 2, 1? would it improve reconstruction and linear probing results?
 - what is the effect of decreasing batch size?
 - experiment with regularisation, especially L1 and L2 on h
 - add dropout
 - lower weight updates to 0.0001
 - experiment with converging up to a certain energy during inference, instead for a fixed number of steps
 - experiment with muzero like regularisation as planned. start small. commit first initial probing results.

Current state:
  - for 6block model even with the close to all time best final train mse 0.004531 the linear probing results are merely 0.18 for vode_0
  - something might be off with the feature creation, what if we have some catastrophic collapse of the hidden states? when I set the actual LR that was used when the model was extracted, then the image generated gets very bad. the reference video is generated after the feature extraction, so that proces already messed with the latents significantly. So,  I mean, if we reset h during training, and if we did that here as well, that is the only way we can ensure this does not happen. but I have not experimented with that setup for a long time. but, that setup could also ensure perhaps that we get better higher latents? it might force the network, to always try to find good representations from scratch, and not just rely on the weights. 


Commands:
Block6:

 python linear_probe.py --config_name 6block --feature_layer_vode_indices "0,1,2,3,4,5,6,7" --concatenate_features True --probe_inference_steps 20 --probe_h_lr 0.048514  --model_path ../results/models/nb6_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd80_epoch69_trainmse0.004531_20250523_173804.npz

  python linear_probe.py --config_name 6block --feature_layer_vode_indices "0" --concatenate_features True --probe_inference_steps 20 --probe_h_lr 0.095  --model_path ../results/models/nb6_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd10_epoch75_trainmse0.003547_20250526_155633.npz

  python linear_probe.py --config_name 6block --probe_inference_steps 20 --probe_h_lr 0.095 --model_path ../results/models/nb6_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd10_epoch75_trainmse0.003547_20250526_155633.npz

  python linear_probe.py --config_name 6block --feature_layer_vode_indices "0,1,2,3,4,5,6,7" --concatenate_features True --probe_inference_steps 20 --probe_h_lr 0.095 --model_path ../results/models/nb6_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd10_epoch75_trainmse0.003547_20250526_155633.npz

  python linear_probe.py --config_name 6block --feature_layer_vode_indices "0, 6, 7" --concatenate_features True --probe_inference_steps 100 --probe_h_lr 0.095 --seed 10 --model_path ../results/models/nb6_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd10_epoch75_trainmse0.003547_20250526_155633.npz

Block 1:
 python linear_probe.py --config_name 1block --feature_layer_vode_indices "0" --concatenate_features True --probe_inference_steps 20 --probe_h_lr 0.09351  --model_path ../results/models/nb1_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e150_sd80_epoch18_finalabstrainmse0.000436_20250524_100129.npz

  python linear_probe.py --config_name 1block --feature_layer_vode_indices "0" --concatenate_features True --probe_inference_steps 20 --probe_h_lr 0.048514  --model_path ../results/models/nb1_bs500_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e75_sd80_epoch69_trainmse0.003560_20250524_120732.npz
 

Block 2:
 python linear_probe.py --config_name 2block --feature_layer_vode_indices "0" --concatenate_features True --probe_inference_steps 20 --probe_h_lr 0.094372 --model_path ../results/models/nb2_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e150_sd80_epoch12_trainmse0.002394_20250524_100558.npz

Block 3:
 python linear_probe.py --config_name 3block --feature_layer_vode_indices "0" --concatenate_features True --probe_inference_steps 20 --probe_h_lr 0.09333 --model_path ../results/models/nb3_bs200_hs64_nh1_lrh0p095_sb1p25_is20_ws0_hm0p40_hclip2000_wclip500_vlnOFF_e150_sd80_epoch19_trainmse0.002849_20250524_101515.npz