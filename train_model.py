import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os


class ImprovedHandModel(nn.Module):
    """Improved neural network model for hand opening prediction"""

    def __init__(self, input_size=47, hidden_size=256):  # 42 coordinates + 5 features = 47
        super(ImprovedHandModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
            # No Sigmoid - use raw output for better regression performance
        )

    def forward(self, x):
        return self.network(x)


class ImprovedHandModelTrainer:
    def __init__(self, csv_file='improved_hand_data.csv'):
        self.csv_file = csv_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = None
        self.scaler = StandardScaler()

        # Enhanced hyperparameters
        self.batch_size = 16  # Smaller batch size for better training
        self.learning_rate = 0.0005  # Lower learning rate for stability
        self.epochs = 200  # More epochs for convergence
        self.validation_split = 0.2
        self.early_stopping_patience = 30

    def load_data(self):
        """Load and prepare enhanced data from CSV"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"File {self.csv_file} does not exist. Run data collection first.")

        print(f"Loading enhanced data from {self.csv_file}...")
        df = pd.read_csv(self.csv_file)

        print(f"Loaded {len(df)} samples with enhanced features")
        
        # Analyze data distribution
        print(f"Target angle distribution:")
        print(df['target_angle'].value_counts().sort_index())
        
        # Check if we have sufficient data
        if len(df) < 20:
            print("⚠️  Warning: Very little training data. Recommended minimum: 100+ samples.")
        elif len(df) < 50:
            print("⚠️  Warning: Limited training data. Consider collecting more samples.")

        # Prepare input features (all columns except target_angle)
        feature_columns = [col for col in df.columns if col != 'target_angle']
        X = df[feature_columns].values
        y = df['target_angle'].values

        # Check for NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            print("⚠️  Detected NaN values - cleaning data...")
            df_clean = df.dropna()
            X = df_clean[feature_columns].values
            y = df_clean['target_angle'].values
            print(f"After cleaning: {len(X)} samples remain")

        # Validate expected input size
        expected_features = 47  # 42 coordinates + 5 calculated features
        if X.shape[1] != expected_features:
            print(f"⚠️  Warning: Expected {expected_features} features, got {X.shape[1]}")
            print("This might indicate a mismatch between data collection and training scripts")

        # Information about data size
        print(f"Input data shape: {X.shape}")
        print(f"Output data shape: {y.shape}")
        
        # Feature analysis
        print(f"\nFeature ranges:")
        feature_names = ['thumb_index_dist', 'thumb_angle', 'index_angle', 'hand_size', 'normalized_dist']
        if len(feature_columns) >= 5:
            for i, name in enumerate(feature_names):
                if i < len(feature_columns) - 42:  # Last 5 features
                    col_idx = 42 + i  # Skip coordinate columns
                    if col_idx < len(feature_columns):
                        print(f"{name}: {X[:, col_idx].min():.4f} to {X[:, col_idx].max():.4f}")

        return X, y

    def prepare_data(self, X, y):
        """Prepare data for training with improved preprocessing"""
        # Stratified split to ensure balanced representation
        # For regression, we'll use binning for stratification
        y_binned = pd.cut(y, bins=5, labels=False)
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=42, 
                stratify=y_binned
            )
        except ValueError:
            # Fallback to regular split if stratification fails
            print("Using regular train/test split (stratification failed)")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=42
            )

        # Enhanced normalization
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Keep target values in original range [0, 100] for better training
        # No normalization of target values - direct regression

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

        # DataLoaders with shuffle for better training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Training data: {len(X_train)} samples")
        print(f"Validation data: {len(X_val)} samples")
        print(f"Training target range: {y_train.min():.1f}% to {y_train.max():.1f}%")
        print(f"Validation target range: {y_val.min():.1f}% to {y_val.max():.1f}%")

        return train_loader, val_loader, X_val, y_val

    def train_model(self, train_loader, val_loader):
        """Train the improved model with advanced techniques"""
        # Initialize model
        input_size = 47  # 42 coordinates + 5 enhanced features
        self.model = ImprovedHandModel(input_size=input_size).to(self.device)

        # Advanced optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, verbose=True
        )
        
        # Loss function - MSE for regression
        criterion = nn.MSELoss()

        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Enhanced model with {input_size} input features")
        print(f"Architecture: {input_size} -> 256 -> 128 -> 64 -> 32 -> 1")

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batches += 1

            # Calculate average losses
            avg_train_loss = train_loss / max(train_batches, 1)
            avg_val_loss = val_loss / max(val_batches, 1)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'improved_hand_model_best.pth')
            else:
                patience_counter += 1

            # Display progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"LR: {current_lr:.6f}")

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if os.path.exists('improved_hand_model_best.pth'):
            self.model.load_state_dict(torch.load('improved_hand_model_best.pth'))
            print("Loaded best model from training")

        return train_losses, val_losses

    def evaluate_model(self, X_val, y_val):
        """Evaluate model with comprehensive metrics"""
        self.model.eval()

        X_val_scaled = self.scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_val_tensor).cpu().numpy().flatten()

        # Clamp predictions to valid range
        predictions = np.clip(predictions, 0, 100)

        # Comprehensive metrics
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        rmse = np.sqrt(np.mean((y_val - predictions) ** 2))
        
        # Additional metrics
        mape = np.mean(np.abs((y_val - predictions) / np.maximum(y_val, 1))) * 100  # Avoid division by zero
        max_error = np.max(np.abs(y_val - predictions))

        print(f"\n=== IMPROVED MODEL EVALUATION ===")
        print(f"Mean Absolute Error (MAE): {mae:.2f}%")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f}%")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Maximum Error: {max_error:.2f}%")

        # Detailed error analysis
        errors = np.abs(y_val - predictions)
        print(f"\nError Analysis:")
        print(f"• Errors < 5%: {np.sum(errors < 5) / len(errors) * 100:.1f}% of samples")
        print(f"• Errors < 10%: {np.sum(errors < 10) / len(errors) * 100:.1f}% of samples")
        print(f"• Errors < 15%: {np.sum(errors < 15) / len(errors) * 100:.1f}% of samples")

        # Sample predictions
        print(f"\nSample predictions:")
        indices = np.random.choice(len(y_val), min(10, len(y_val)), replace=False)
        for i in indices:
            error = abs(y_val[i] - predictions[i])
            print(f"True: {y_val[i]:5.1f}% | Pred: {predictions[i]:5.1f}% | Error: {error:4.1f}%")

        return predictions, mae, r2, rmse

    def plot_training_history(self, train_losses, val_losses):
        """Plot comprehensive training history"""
        plt.figure(figsize=(15, 10))

        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.title('Training History - Enhanced Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Loss curves (log scale)
        plt.subplot(2, 3, 2)
        plt.semilogy(train_losses, label='Training Loss', alpha=0.8)
        plt.semilogy(val_losses, label='Validation Loss', alpha=0.8)
        plt.title('Training History (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE, log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning curve analysis
        plt.subplot(2, 3, 3)
        if len(train_losses) > 10:
            smooth_window = min(10, len(train_losses) // 5)
            train_smooth = pd.Series(train_losses).rolling(smooth_window).mean()
            val_smooth = pd.Series(val_losses).rolling(smooth_window).mean()
            plt.plot(train_smooth, label=f'Training (MA-{smooth_window})', alpha=0.8)
            plt.plot(val_smooth, label=f'Validation (MA-{smooth_window})', alpha=0.8)
        plt.title('Smoothed Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Model architecture info
        plt.subplot(2, 3, 4)
        plt.text(0.1, 0.9, 'Enhanced Model Architecture:', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.8, '• Input: 47 features', fontsize=10)
        plt.text(0.1, 0.7, '  - 42 hand coordinates', fontsize=10)
        plt.text(0.1, 0.6, '  - 5 calculated features', fontsize=10)
        plt.text(0.1, 0.5, '• Hidden layers: 256→128→64→32', fontsize=10)
        plt.text(0.1, 0.4, '• Dropout, BatchNorm', fontsize=10)
        plt.text(0.1, 0.3, '• Output: Hand opening %', fontsize=10)
        plt.text(0.1, 0.2, '• AdamW + LR Scheduler', fontsize=10)
        plt.text(0.1, 0.1, '• Early Stopping', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # Training statistics
        plt.subplot(2, 3, 5)
        final_train_loss = train_losses[-1] if train_losses else 0
        final_val_loss = val_losses[-1] if val_losses else 0
        min_val_loss = min(val_losses) if val_losses else 0
        
        plt.text(0.1, 0.9, 'Training Statistics:', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.8, f'Epochs completed: {len(train_losses)}', fontsize=10)
        plt.text(0.1, 0.7, f'Final train loss: {final_train_loss:.4f}', fontsize=10)
        plt.text(0.1, 0.6, f'Final val loss: {final_val_loss:.4f}', fontsize=10)
        plt.text(0.1, 0.5, f'Best val loss: {min_val_loss:.4f}', fontsize=10)
        plt.text(0.1, 0.4, f'Device: {self.device}', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # Feature importance placeholder
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.5, 'Enhanced Features:\n\n• Thumb-index distance\n• Finger angles\n• Hand size normalization\n• Multi-finger distances\n• Geometric features',
                 ha='center', va='center', fontsize=10)
        plt.title('Feature Enhancements')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('improved_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Training history saved as 'improved_training_history.png'")

    def plot_predictions(self, y_true, y_pred):
        """Plot comprehensive prediction analysis"""
        plt.figure(figsize=(16, 12))

        # Predictions vs True values
        plt.subplot(3, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, s=30)
        plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('True Values (%)')
        plt.ylabel('Predictions (%)')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Residuals plot
        plt.subplot(3, 3, 2)
        residuals = y_pred - y_true
        plt.scatter(y_true, residuals, alpha=0.6, s=30)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Values (%)')
        plt.ylabel('Residuals (%)')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)

        # Error histogram
        plt.subplot(3, 3, 3)
        errors = np.abs(y_pred - y_true)
        plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error (%)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)

        # True values distribution
        plt.subplot(3, 3, 4)
        plt.hist(y_true, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('True Values (%)')
        plt.ylabel('Frequency')
        plt.title('True Values Distribution')
        plt.grid(True, alpha=0.3)

        # Predictions distribution
        plt.subplot(3, 3, 5)
        plt.hist(y_pred, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Predictions (%)')
        plt.ylabel('Frequency')
        plt.title('Predictions Distribution')
        plt.grid(True, alpha=0.3)

        # Box plot comparison
        plt.subplot(3, 3, 6)
        plt.boxplot([y_true, y_pred], labels=['True', 'Predicted'])
        plt.ylabel('Values (%)')
        plt.title('Distribution Comparison')
        plt.grid(True, alpha=0.3)

        # Error by true value
        plt.subplot(3, 3, 7)
        plt.scatter(y_true, errors, alpha=0.6, s=30, c='red')
        plt.xlabel('True Values (%)')
        plt.ylabel('Absolute Error (%)')
        plt.title('Error vs True Value')
        plt.grid(True, alpha=0.3)

        # Cumulative error distribution
        plt.subplot(3, 3, 8)
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        plt.plot(sorted_errors, cumulative)
        plt.xlabel('Absolute Error (%)')
        plt.ylabel('Cumulative Percentage')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)

        # Performance metrics text
        plt.subplot(3, 3, 9)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs(residuals / np.maximum(y_true, 1))) * 100
        
        metrics_text = f"""Performance Metrics:

MAE: {mae:.2f}%
RMSE: {rmse:.2f}%
R²: {r2:.4f}
MAPE: {mape:.2f}%

Accuracy within:
±5%: {np.sum(errors < 5)/len(errors)*100:.1f}%
±10%: {np.sum(errors < 10)/len(errors)*100:.1f}%
±15%: {np.sum(errors < 15)/len(errors)*100:.1f}%"""

        plt.text(0.1, 0.9, metrics_text, fontsize=10, verticalalignment='top')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('improved_model_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Model evaluation plots saved as 'improved_model_evaluation.png'")

    def save_model(self):
        """Save the trained model and scaler"""
        # Save the model
        torch.save(self.model.state_dict(), 'improved_hand_model.pth')
        
        # Save the scaler
        joblib.dump(self.scaler, 'improvedd_scaler.pkl')

        # Save model info
        model_info = {
            'input_size': 47,
            'model_architecture': 'Enhanced hand opening predictor',
            'features': '42 coordinates + 5 calculated features',
            'training_samples': 'varies',
            'version': 'improved_v1.0'
        }
        
        with open('improved_model_info.txt', 'w') as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")

        print(f"\n✓ Enhanced model saved as 'improved_hand_model.pth'")
        print(f"✓ Scaler saved as 'improved_scaler.pkl'")
        print(f"✓ Model info saved as 'improved_model_info.txt'")
        print(f"✓ Model ready for inference with 47 input features")

    def analyze_data_distribution(self, df):
        """Analyze enhanced training data distribution"""
        print(f"\n=== ENHANCED DATA ANALYSIS ===")
        
        print("Target angle distribution:")
        angles = df['target_angle'].value_counts().sort_index()
        print(angles)
        
        print(f"\nDataset statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Mean angle: {df['target_angle'].mean():.1f}%")
        print(f"Median angle: {df['target_angle'].median():.1f}%")
        print(f"Std deviation: {df['target_angle'].std():.1f}%")
        print(f"Range: {df['target_angle'].min():.1f}% - {df['target_angle'].max():.1f}%")

        # Analyze enhanced features if they exist
        feature_cols = ['thumb_index_dist', 'thumb_angle', 'index_angle', 'hand_size', 'normalized_dist']
        for col in feature_cols:
            if col in df.columns:
                print(f"{col}: {df[col].min():.4f} to {df[col].max():.4f} (mean: {df[col].mean():.4f})")

    def train_full_pipeline(self):
        """Complete training pipeline for enhanced model"""
        try:
            print("=== ENHANCED HAND MODEL TRAINING ===\n")
            
            # Load enhanced data
            X, y = self.load_data()
            
            # Analyze data distribution
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                self.analyze_data_distribution(df)

            # Prepare data
            train_loader, val_loader, X_val, y_val = self.prepare_data(X, y)

            # Train model
            train_losses, val_losses = self.train_model(train_loader, val_loader)

            # Evaluate model
            predictions, mae, r2, rmse = self.evaluate_model(X_val, y_val)

            # Generate plots
            self.plot_training_history(train_losses, val_losses)
            self.plot_predictions(y_val, predictions)

            # Save model
            self.save_model()

            # Final summary
            print(f"\n🎉 ENHANCED TRAINING COMPLETED!")
            print(f"📊 Final metrics: MAE = {mae:.2f}%, RMSE = {rmse:.2f}%, R² = {r2:.4f}")
            print(f"📈 Model trained on {len(X)} samples with enhanced features")
            print(f"🚀 Ready for inference with improved accuracy!")

            # Performance assessment
            if mae < 10:
                print("🟢 Excellent performance! MAE < 10%")
            elif mae < 15:
                print("🟡 Good performance! MAE < 15%")
            else:
                print("🔴 Consider collecting more training data or tuning hyperparameters")

            return True

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    print("=== ENHANCED HAND OPENING MODEL TRAINING ===\n")
    print("This script trains an improved model with enhanced features:")
    print("• 42 hand landmark coordinates")
    print("• 5 calculated geometric features")
    print("• Advanced neural network architecture")
    print("• Improved training techniques\n")

    trainer = ImprovedHandModelTrainer()
    success = trainer.train_full_pipeline()

    if success:
        print("\n🚀 ENHANCED MODEL READY FOR USE!")
        print("Run inference with: python hand_analyze.py --mode inference")
        print("\n💡 Model improvements:")
        print("• Enhanced feature engineering")
        print("• Better architecture with BatchNorm and Dropout")
        print("• Advanced training with early stopping")
        print("• Comprehensive evaluation metrics")
        print("• Improved data preprocessing")
    else:
        print("\n💡 Troubleshooting suggestions:")
        print("• Ensure you ran data collection first:")
        print("  python hand_analyze.py --mode collect")
        print("• Check that the CSV file has the correct format")
        print("• Try collecting more diverse training data")


if __name__ == "__main__":
    main()