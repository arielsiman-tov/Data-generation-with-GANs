# Data-generation-with-GANs
The goal of this project is to train and compare a Generative Adversarial Network (GAN) and a Conditional GAN (cGAN) on the “Adult” dataset, then evaluate the similarity between real and synthetic samples as well as the usefulness of the synthetic data as a substitute for real data. 


### We followed these stages in our work:

1.	Load and preprocess the “Adult” dataset (in ARFF format).
2.	Perform an 80%/20% train-test split, preserving label ratios, repeated with three different random seeds.
3.	Train both a standard GAN and a conditional GAN (cGAN).
4.	Visualize and evaluate the performance using two key metrics:   
   * Detection (how easily a classifier can distinguish real vs. synthetic data),
   * Efficacy (how well a model trained on synthetic data performs on real data).

### Dataset and Preprocessing:

Dataset: The Adult dataset (also known as the Census Income dataset) contains demographic information such as age, workclass, education, marital status, occupation, race, gender, hours-per-week, native country, etc., along with a binary label “income” (>50K or ≤50K).

Steps:
1.	Train-Test Split: The data was split into 80% for training and 20% for testing, ensuring we maintained the class distribution. This step was repeated for three different random seeds {12,21,27}.
2.	Handling Categorical Variables: Used one-hot encoding (OneHotEncoder) on all categorical features. (We expanded our data representation from 14 features to 108, excluding the target column.)
3.	Handling Numerical Variables: Scaled continuous features using MinMaxScaler with a range of [−1,+1].
4.	Label Encoding: Converted the binary label into 0 for “≤50K” and 1 for ">50K”.
5.	Final Shape: After preprocessing, the dataset was represented as a NumPy array X_processed for y_encoded for labels.

### Models and Architectures:

### We implemented two main architectures: a standard GAN and a conditional GAN (cGAN). For both:
•	Generator: Takes random noise (and optionally conditional labels in cGAN) as input and outputs synthetic samples of the same dimensionality as the real data.
•	Discriminator: Attempts to distinguish between real and synthetic samples (and also uses labels in cGAN).
cGAN - Similar to the standard GAN generator but with additional embedding and concatenation of class information.

### GAN/cGAN with AutoEncoder:
We trained a separate autoencoder architecture, then use the encoder component to create an embedding of the real sample before it is sent to the discriminator. We tried this improvement for both models. 

### Usage of the Autoencoder (AE) in GAN Training:
•	Optional Encoding Step: The code includes a section where the real data (data_row) could be encoded using an autoencoder before training the GAN.
•	Purpose of the Autoencoder: If enabled, the autoencoder would transform real data into a lower-dimensional latent space, making the GAN operate on learned feature representations instead of raw data. 

With AE, the GAN generator might learn to produce meaningful embeddings that are later decoded into the original space, whereas without AE, it directly synthesizes data from random noise in the original feature space.

Using the autoencoder may improve stability and reduce the complexity of modeling high-dimensional data, potentially leading to better quality synthetic samples. Later in this work, we will analyze the impact of adding AE to our model architecture.

### Training Setup
#### Hyperparameters
•	Batch Size: 128
•	Noise Dimension (nz): 128
•	Learning Rate: 0.0001
•	Weight Decay: 0.00005
•	Optimizer: Adam (both Generator and Discriminator)
•	Loss Function: Binary Cross-Entropy (BCELoss)
•	Number of Epochs: both Standard GAN and cGAN: 50 epochs

#### Training Procedure
##### GAN:

Discriminator Update:
* Sample a batch of real data from the training set.
* Generate a batch of fake data (via the Generator).
* Compute discriminator loss for real vs. fake samples, backpropagate, and update the discriminator weights.

Generator Update:
* Generate fake data.
* Pass it through the discriminator and compute the generator loss, which seeks to fool the discriminator into classifying fake data as real.
* Backpropagate and update generator weights.
##### cGAN:
The same structure as above, but the generator and discriminator both receive label information as additional input.

#### Stabilization Techniques
*	LeakyReLU: Used in generator and discriminator to avoid dying ReLU problems.
*	Dropout: Added discriminator layers to mitigate overfitting and improve training stability.
*	Label Embeddings: For the cGAN, use nn.Embedding to embed the label into a dense vector space.

### A Comparison between the loss values of GAN vs cGAN:

#### Generator loss 

![image](https://github.com/user-attachments/assets/c267d8dc-dffb-4a04-8eb5-c48744ffeede)

#### Discriminator loss 

![image](https://github.com/user-attachments/assets/78199077-b125-46dc-80ee-bc730ea783a2)

We can see that in both models the generator's loss increases gradually, while the discriminator's loss decreases, indicating that the discriminator becomes better at distinguishing real from fake samples as training progresses. The increasing generator loss suggests it struggles to produce convincing samples as the discriminator improves. The smoothed trends indicate some stability over time, with neither loss collapsing, suggesting that the GAN avoids major training instabilities.
#### The GAN model exhibits slightly improved loss values compared to the cGAN model.

#### Feature analysis cGAN
Comparing the feature distributions of the synthetic data vs the original data – numeric features

![image](https://github.com/user-attachments/assets/cce8b119-28c7-403a-9109-ed51fa1efbb8)

These plots reveal differences between real and synthetic data distributions across several features. 
For features like age, fnlwgt, capital-gain and capital-loss, the synthetic data captures the overall trends but deviates in certain peaks and variances, indicating an incomplete replication of the real data's nuances. 
For education-num and hours-per-week, the synthetic data shows less accuracy in replicating the distribution of the original data, with noticeable deviations in key peaks and overall patterns.

#### Comparing correlation matrices 

![image](https://github.com/user-attachments/assets/4181f10e-ea2f-4af7-ac28-046761618aa4)

The correlation matrices for real and synthetic datasets reveal significant differences in how relationships between numerical variables are captured. In the real data, weak correlations are evident between variables such as capital-gain and capital-loss, indicating that these variables are not linked or influenced by similar factors. In contrast, the synthetic data struggles to replicate these relationships accurately. While some correlations are preserved, such as between age and hours-per-week, other relationships, like those involving capital-gain and capital-loss, are represented wrongly.

#### Comparing the feature distributions of the synthetic data vs the original data – categorial features

![image](https://github.com/user-attachments/assets/05254dd2-e8c4-4e1e-b28e-510b030a276c)

For attributes like workclass, the synthetic data tends to overrepresent common categories such as "Private" or "local-gov", while underrepresenting less frequent ones like "Self-employed." This discrepancy may be due to the synthetic model's difficulty in balancing rare and frequent classes. 
Similarly, in education, the synthetic dataset fails to accurately replicate rare categories such as "Doctorate” or some-college", indicating that the model struggles to handle infrequent data points, likely due to insufficient examples during training.
For attribute like sex the synthetic data aligns relatively well with the real data, likely because of its binary nature, which is simpler to replicate.
The native-country attribute reveals the greatest divergence, with the synthetic data poorly representing many countries and overemphasizing the majority. This is likely due to the high imbalance and the large number of unique values, which make accurate replication challenging. 

#### we did  the same feature analysis for the 4 combinations: (1) GAN, (2) cGAN, (3) GAN wit AE, (4) cGAN with AE (attached in the report)

### We have conducted an experiment in which we incorporated an Autoencoder (AE) into the models' architecture: 

#### loss values of cGAN with AE
Generator loss 

![image](https://github.com/user-attachments/assets/0853cbde-56ff-41c6-817c-50ee5092352d)

#### Discriminator loss

![image](https://github.com/user-attachments/assets/647994c2-cff6-4bc4-b185-0e376e5d726e)

A notable change occurs around step 2,000-3,000, where both losses stabilize after initial volatility. 
The Generator's loss starts high (around 3.0-3.5) before settling at approximately 1.67, while the Discriminator's loss initially fluctuates between 0.4-1.2 before stabilizing around 0.9. The convergence of both losses to relatively stable values suggest the GAN reached a reasonable equilibrium in its training, though the continuing small oscillations indicate the ongoing adversarial nature of the training process.

#### loss values of GAN with AE
Generator loss 

![image](https://github.com/user-attachments/assets/0d0d3f51-ab52-4dc5-95b0-487040aac160)

Discriminator loss

![image](https://github.com/user-attachments/assets/1127433b-2c03-4ff2-9ef4-b19fe9d87793)

In this case the key transition also occurs around step 2,000-3,000, where both losses stabilize. The Generator's loss peaks at around 4.0 before settling to approximately 1.9, while the Discriminator's loss starts with high volatility (peaking around 1.2-1.4) before stabilizing around 0.7.
Compared to the previous plots, the overall pattern and convergence behavior is very similar, suggesting no fundamental change in the training dynamics. The differences are minor and within normal variation for GAN training. Both runs appear to have reached a stable equilibrium, and the similar patterns suggest consistent, reproducible training behavior, although the cGAN with AE model achieved the best results in terms of loss function performance. 

### Summary of results:

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/03f42118-4c0e-4d9b-9508-33db38a0226a" />

#### The results show that adding an Autoencoder (AE) significantly impacts both the generator and discriminator losses. 
Without AE, the cGAN has higher generator and discriminator losses compared to the GAN. However, with AE, both models achieve considerably lower generator losses, with the cGAN showing the lowest at 1.67. The discriminator losses increase when AE is added, with the cGAN showing the highest discriminator loss (0.91). This suggests that incorporating AE helps improve the generator's performance while making the discriminator's task more challenging.

### Evaluation: To evaluate the results, we employed two key metrics: Detection and Efficacy.

#### Detection Metric - The Detection metric evaluates how well the synthetic data mimics the real data.
* Combines real and synthetic data in a 50/50 mix.
* Trains a Random Forest on three folds of each data type (real/synthetic), then tests on the remaining fold (again, containing 50% real, 50% synthetic).
* The average AUC (Area Under the ROC Curve) measures how well the Random Forest can distinguish real from synthetic. Lower AUC indicates better similarity.
 
#### Efficacy Metric - the Efficacy metric measures the utility of synthetic data as a substitute for real data in training predictive models. It consists of:
* Training a Random Forest on real training data and evaluating on real test data to obtain auc_real_efficacy.
* Training a Random Forest on synthetic training data and evaluating on real test data to obtain auc_synth_efficacy.
* Computing the Efficacy Ratio: Efficacy Ratio = (auc synth efficacy)  \(auc real efficacy). a ratio close to 1 indicates the synthetic data closely matches the utility of real data. 

#### cGAN efficiency results

![image](https://github.com/user-attachments/assets/75568b23-75fd-4b91-9770-2c49c11535aa)

#### GAN efficiency results

![image](https://github.com/user-attachments/assets/dd9fd566-cfac-4005-b8b4-948e043fdaf4)

#### cGAN with AE efficiency results

![image](https://github.com/user-attachments/assets/64cc4018-839e-490f-923b-a403554555ba)

#### GAN with AE efficiency results

![image](https://github.com/user-attachments/assets/ddc4e779-1eed-40f7-9f2b-6c016f01f549)

### Key findings:
* GAN and cGAN: Both achieve similar detection performance, indicating bad abilities to mimic real data, because the average AUC Detection is 1. 
Both show moderate efficacy ratios, demonstrating that synthetic data is useful but not as reliable as the real data for training.
* GAN with AE and cGAN with AE: Adding the Autoencoder (AE) doesn’t improve the detection performance. Adding the AE significantly boosts the efficacy ratio. In particular, cGAN with AE achieves the highest average efficacy ratio (0.62), indicating it produces the most effective synthetic data for model training.

