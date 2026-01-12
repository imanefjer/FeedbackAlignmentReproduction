import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Hyperparameters
LEARNING_RATE = 1e-3 
WEIGHT_DECAY = 1e-6

# Architecture
INPUT_DIM = 784
HIDDEN_DIM = 1000
OUTPUT_DIM = 10

# Training settings
BATCH_SIZE = 64 
EPOCHS = 100    

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# Set deterministic seed
SEED = 2014 
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Data Loading (MNIST)
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    return train_loader, test_loader

# The Model (Manual Implementation)
class Network:
    def __init__(self, mode='bp'):
        """
        mode: 'bp' (Backprop), 'fa' (Feedback Alignment), 'shallow'
        """
        self.mode = mode
        
        # --- Initialization (Page 10: Full Methods) ---
        # "Elements of W0 and W were drawn from uniform distribution over [-w, w]"
        # "Elements of B were drawn from uniform distribution over [-b, b]"
        # The paper performed manual search. 
        # We use Xavier-like scaling which is the result of such searches.
        
        w_scale = np.sqrt(6.0 / (INPUT_DIM + HIDDEN_DIM))
        self.W0 = torch.empty(INPUT_DIM, HIDDEN_DIM, device=DEVICE).uniform_(-w_scale, w_scale)
        self.b0 = torch.zeros(1, HIDDEN_DIM, device=DEVICE)
        
        w1_scale = np.sqrt(6.0 / (HIDDEN_DIM + OUTPUT_DIM))
        self.W = torch.empty(HIDDEN_DIM, OUTPUT_DIM, device=DEVICE).uniform_(-w1_scale, w1_scale)
        self.b = torch.zeros(1, OUTPUT_DIM, device=DEVICE)
        
        # The Fixed Random Matrix B for FA
        # Page 9: "drawn from uniform distribution... scale chosen by manual search"
        # Empirically, B needs to be roughly the same scale as the forward weights to work well.
        self.B = torch.empty(OUTPUT_DIM, HIDDEN_DIM, device=DEVICE).uniform_(-0.5, 0.5)

        # Storage for updates to calculate angle
        self.bp_update = None
        self.fa_update = None

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def d_sigmoid(self, s):
        # Derivative of sigmoid with respect to input, given output s
        # sigma'(x) = sigma(x) * (1 - sigma(x))
        return s * (1.0 - s)

    def forward(self, x):
        self.x = x
        
        # Hidden Layer
        self.a0 = x.mm(self.W0) + self.b0
        self.h = self.sigmoid(self.a0)
        
        # Output Layer
        self.a1 = self.h.mm(self.W) + self.b
        self.y = self.sigmoid(self.a1) 
        
        return self.y

    def train_step(self, target):
        """
        Manual backward pass to implement Eq 2: Delta_h = B e
        """
        batch_size = target.shape[0]
        
        # 1. Error Calculation
        # Paper: e = y* - y. 
        # We want to minimize L = 1/2 e^T e.
        # dL/dy = -(y* - y) = -e.
        # However, we are passing through the output non-linearity (Sigmoid).
        # delta_out = dL/da1 = (dL/dy) * (dy/da1) 
        # delta_out = -(target - y) * y(1-y)
        
        diff = target - self.y # (y* - y)
        d_sigma_out = self.d_sigmoid(self.y)
        delta_out = -diff * d_sigma_out # The gradient signal at the output logits
        
        # NOTE: The paper defines update rules as proportional to (-gradient).
        # So we work with the negative gradient (the direction we want to move).
        # Let 'e_signal' be (target - y) * sigma'
        e_signal = diff * d_sigma_out

        # 2. Compute Updates for Output Weights (W)
        # dW = x^T * delta
        # Update: W += learning_rate * (h^T * e_signal)
        # We use SUM of gradients (implicit in learning rate magnitude 1e-3 from 2014)
        dW = self.h.t().mm(e_signal)
        db = torch.sum(e_signal, dim=0, keepdim=True)

        # 3. Compute Error Signal for Hidden Layer
        
        # --- Backprop Path (Reference) ---
        # delta_h_bp = (e_signal * W^T) * sigma'(h)
        # This is what standard backprop does.
        err_bp = e_signal.mm(self.W.t())
        d_sigma_h = self.d_sigmoid(self.h)
        delta_h_bp = err_bp * d_sigma_h
        self.bp_update = delta_h_bp # Store for angle calc

        # --- Feedback Alignment Path ---
        # Page 4 Eq 2: "delta_h = B e" (conceptually)
        # Page 9: "Delta h_FA = (B e) o sigma'"
        # Here 'e' refers to the error signal coming from the layer above.
        
        if self.mode == 'bp':
            delta_h = delta_h_bp
        elif self.mode == 'fa':
            # Use fixed B matrix instead of W.T
            # err_fa = e_signal * B
            err_fa = e_signal.mm(self.B) 
            delta_h = err_fa * d_sigma_h
            self.fa_update = delta_h # Store for angle calc
        elif self.mode == 'shallow':
            delta_h = torch.zeros_like(self.h)
            self.fa_update = torch.zeros_like(self.h)

        # 4. Compute Updates for Hidden Weights (W0)
        dW0 = self.x.t().mm(delta_h)
        db0 = torch.sum(delta_h, dim=0, keepdim=True)

        # 5. Apply Updates (SGD + Weight Decay)
        # Paper Page 9: "Both algorithms used learning rate n=10^-3 and weight decay a=10^-6"
        
        # W update
        self.W  += LEARNING_RATE * dW  - (LEARNING_RATE * WEIGHT_DECAY) * self.W
        self.b  += LEARNING_RATE * db  - (LEARNING_RATE * WEIGHT_DECAY) * self.b
        
        # W0 update
        if self.mode != 'shallow':
            self.W0 += LEARNING_RATE * dW0 - (LEARNING_RATE * WEIGHT_DECAY) * self.W0
            self.b0 += LEARNING_RATE * db0 - (LEARNING_RATE * WEIGHT_DECAY) * self.b0

    def get_angle(self):
        if self.mode != 'fa': return 0.0
        
        # Calculate angle between the prescribed BP update and the actual FA update
        # Vectors are flattened over the batch
        v_bp = self.bp_update.reshape(-1)
        v_fa = self.fa_update.reshape(-1)
        
        # Cosine similarity
        dot = torch.dot(v_bp, v_fa)
        norm_bp = torch.norm(v_bp)
        norm_fa = torch.norm(v_fa)
        
        if norm_bp == 0 or norm_fa == 0: return 90.0
        
        cos_theta = dot / (norm_bp * norm_fa)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        angle = torch.acos(cos_theta)
        return torch.rad2deg(angle).item()

# Experiment Runner
def run_experiment(mode, train_loader, test_loader):
    print(f"\nTraining {mode.upper()}...")
    net = Network(mode=mode)
    
    # 1-hot encoder
    eye = torch.eye(10, device=DEVICE)
    
    errors = []
    angles = []
    
    correct = 0
    total = 0
    with torch.no_grad(): 
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            output = net.forward(imgs)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    initial_error = 100.0 * (1.0 - correct/total)
    errors.append(initial_error)
    angles.append(90.0 if mode == 'fa' else 0.0) 
    
    print(f"Epoch 0 | Error: {initial_error:.2f}% | (Before Training)")

    for epoch in range(EPOCHS):
        epoch_angles = []
        
        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            imgs = imgs.to(DEVICE)
            targets = eye[labels]
            
            # Forward
            net.forward(imgs)
            
            # Train step (Manual Backward)
            net.train_step(targets)
            
            if mode == 'fa':
                epoch_angles.append(net.get_angle())
                
        # Evaluate
        correct = 0
        total = 0
        
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            output = net.forward(imgs)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        error_rate = 100.0 * (1.0 - correct/total)
        errors.append(error_rate)
        
        avg_angle = np.mean(epoch_angles) if epoch_angles else 0.0
        angles.append(avg_angle)
        
        print(f"Epoch {epoch+1} | Error: {error_rate:.2f}% | Angle: {avg_angle:.1f}")
        
        # Simple Early stopping or learning rate decay if needed (not in paper, but helps convergence)
        # But paper says "fixed learning rate", so we do not decay.
        
    return errors, angles

# Execution
if __name__ == "__main__":
    train, test = load_data()
    
    # Run Backprop
    bp_err, _ = run_experiment('bp', train, test)
    
    # Run Feedback Alignment
    fa_err, fa_ang = run_experiment('fa', train, test)
    
    # Run Shallow
    sh_err, _ = run_experiment('shallow', train, test)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Error Plot
    plt.subplot(1, 2, 1)
    plt.plot(bp_err, 'k-', label='Backprop', linewidth=2)
    plt.plot(fa_err, 'g-', label='Feedback Alignment', linewidth=2)
    plt.plot(sh_err, 'gray', label='Shallow', alpha=0.6)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error (%)')
    plt.title('MNIST Test Error (Fig 2a)')
    plt.legend()
    plt.ylim(0, 15)
    plt.grid(True, alpha=0.3)
    
    # Angle Plot
    plt.subplot(1, 2, 2)
    plt.plot(fa_ang, 'g-', label=r'$\angle \Delta h_{FA}, \Delta h_{BP}$')
    plt.axhline(90, color='k', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Angle (degrees)')
    plt.title('Alignment Angle (Fig 2b)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_2_replication.png', dpi=300)
    print("Saved figure_2_replication.png")
    plt.show()