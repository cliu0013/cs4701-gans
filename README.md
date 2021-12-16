# cs4701-gans

# Introduction
For this project, our group decided to experiment with Generative Adversarial Networks (GANs). GANs are neural networks that are often in computer vision for generating or transforming realistic looking images based on a training dataset. We focused on the (Deep Convolutional) DCGAN architecture, which uses convolutional layers and sets a series of guidelines for how to successfully train GANs. Our goal for this project was to research and understand how DCGANs are so successful at realistic image generation, and investigate how modifying hyperparameters such as optimizer momentum and stochastic gradient descent batch size affect training. In addition, we created our own naive implementation of a GAN for comparison with a DCGAN system.

# References
1. Radford, Alec, Luke Metz, and Soumith Chintala. 2015. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.1511.06434&site=eds-live&scope=site.
2. Shukla, Aditya. “GANs for the Beginner.” Medium. DataX Journal, February 26, 2021. https://medium.com/data-science-community-srm/gans-for-the-beginner-f936504732a0. 
3. Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. “Generative Adversarial Networks.” Communications of the ACM 63 (11): 139–44. doi:10.1145/3422622.
4. Silva, Thalles. Generative Adversarial Network Structure. June 7, 2017. https://sthalles.github.io/intro-to-gans/. 
5. Brock, Andrew, Jeff Donahue, and Karen Simonyan. 2018. “Large Scale GAN Training for High Fidelity Natural Image Synthesis.” https://search.ebscohost.com/login.aspx?direct=true&db=edsarx&AN=edsarx.1809.11096&site=eds-live&scope=site.
6. “Deep Convolutional Generative Adversarial Network.” TensorFlow Core Tutorials. Google, November 17, 2021. https://www.tensorflow.org/tutorials/generative/dcgan.
7. “Kullback–Leibler Divergence,” Wikipedia (Wikimedia Foundation, December 15, 2021), https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Cross_entropy.
8. Hansen, Casper. “Optimizers Explained - Adam, Momentum and Stochastic Gradient Descent.” Machine Learning From Scratch. Machine Learning From Scratch, December 19, 2019. https://mlfromscratch.com/optimizers-explained/#/.
9. Arjovsky, Martin, Soumith Chintala, and Léon Bottou. 2017. “Wasserstein GAN.” https://search-ebscohost-com.proxy.library.cornell.edu/login.aspx?direct=true&db=edsarx&AN=edsarx.1701.07875&site=eds-live&scope=site.
