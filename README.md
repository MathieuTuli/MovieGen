# Movie Gen

Unofficial implementation of Meta's [Movie Gen Models](https://ai.meta.com/static-resource/movie-gen-research-paper/?utm_source=twitter&utm_medium=organic_social&utm_content=thread&utm_campaign=moviegen)

**>** checkout the [todo.md](todo.md) for, a list of my todos

**>** the TAE is done, with outlier loss implemented, and appears to work in reducing latent dots

![gt gif](./readme-media/gt.gif)
![generated gif](./readme-media/gen.gif)

Left is gt, and right is reconstructed, with outlier loss for 10k steps after training without outlier loss for 50k steps, as discussed in the paper. My outlier loss weight is very different cause of overfitting on a smaller resolution gif here and I didn't tune it, so not all the latent dots are gone.

![generated gif](./readme-media/before_outlier.gif)

Here's what it's like without outlier loss: many more artifacts (latent dots?), especially on the left side of the gif.

I'm working on dual 3090s here so this is enough for now to validate the approach works; although  am overfitting so I would like those artifacts to be resolved. But that's a later problem, need to keep momentum for now.

**>** Working on the full MovieGen architecture now. I already implemented most of the Transformer backbone in 2024, so now just adding the bells and whistles for positional/time embeddings, etc. I spent some time to flush out the cfm objective used (see [here](https://gist.github.com/MathieuTuli/b0859a8a62439999a0a33d55cb297189)), so now I'm working on a smaller version of the model that I can run locally to test everything's working. I'm about 90% there, then I'll rent some GPUs to train the TAE and then a small version of MovieGen on a small set of old iPhone videos I scraped, about 1K videos. Once that's working, I'll see how I can maybe provision some more compute and scale this, but that's a couple of months away at least.
