
## Synthetic Data Tests for NR-LSH

These tests perform MIPS on many random vectors. To test the quality of retrieval,
we find the recall. If we want to find the k vectors that result in the largest inner products, the recall is the fraction of our retrieved vectors that
are in the true topk.

### Recall at k.

Using the k_probe_approx function, the following plot shows the recall (z-axis) for all
combinations of k (y-axis) from 1 to 20 and R (x-axis) from 2.4 to 4.0.

![approx synth recall at k](/image/synth_recall_at_k_2.png)

(For some reason, I was unable to add a title or axis labels with the plotting library I was trying. It takes a while to run and I did not save the data! So, just this for now.)

Using the k_probe function. 

![synth recall at k](/images/synth_k_probe_recall_2.png)
