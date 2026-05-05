## Files:

- `pig-rl-model.ipynb`: Reinforcement learning model for thermal control in pig barns using PPO, with 18-dim observations and 5 binary control actions, trained over 3.0M steps.
- `thermal_rgb_stress_inference/pigstress.ipynb`: Classifies pig thermal stress (cold / neutral / hot) from paired RGB and infrared images using a fused `PigStressModel` (ImageNet ResNet18 on RGB with a small IR CNN, concatenated features, 3-way softmax). Trains with 5-fold cross-validation (~5k labeled samples, fold splits from `labels.csv`), optional augmentation, and saves per-fold checkpoints (`pig_stress_model_fold*.pth`); includes validation curves and aggregate confusion matrices. Data paths target a Kaggle-style dataset layout (`RGB/`, `IR/`, `labels.csv`); adjust `DATA_DIR` for local runs.
- `pig_behaviour_classification_old.ipynb`: Legacy behaviour classification model trained on previous dataset (archived).
- `RL_webpage.html`: Presentation of the SwineRL model with closed-loop architecture diagram, evaluation scenarios, and training pipeline documentation.
- `behaviour_classification_farm_data.html`: Visualization of behaviour classification results on farm data (archived).
- `behaviour_classification_old_data.html`: Visualization of behaviour classification results on old dataset (archived).
