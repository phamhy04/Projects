import numpy as np
import os
import matplotlib.pyplot as plt

class NaiveBayes():
    def __init__(self, X, Y, no_classes):
        self.no_samples, self.no_features = X.shape
        self.no_classes = no_classes
        self.eps = 1e-6
    
    def fit(self, X, Y):
        self.class_means = {}
        self.class_var = {}
        self.class_prior = {}
        
        for c in range(self.no_classes):
            Xc = X[Y == c]
            self.class_means[str(c)] = np.mean(Xc, axis = 0)
            self.class_var[str(c)] = np.var(Xc, axis = 0)
            self.class_prior[str(c)] = Xc.shape[0] / self.no_samples
    
    def predict(self, X):
        probs = np.zeros((self.no_samples, self.no_classes))
        
        for c in range(self.no_classes):
            prior = self.class_prior[str(c)]
            likelihood = self.density_func(X, self.class_means[str(c)], self.class_var[str(c)])
            probs[:, c] = np.log(prior) + likelihood
            
        return np.argmax(probs, axis = 1)
    
    def density_func(self, X, mean, sigma):
        #   Calculate probability from Gaussian density function
        const = -(self.no_features/2) * np.log(2*np.pi) - 0.5*np.sum(np.log(sigma + self.eps))
        probs = 0.5 * np.sum(np.square(X - mean)/(sigma + self.eps), axis = 1) 
        return const - probs


if __name__ == "__main__":   
    base_path = 'O:\My_Documents\MACHINE_LEARNING\Datasets'
    X = np.load(os.path.join(base_path, "X.npy"))
    Y = np.load(os.path.join(base_path, "Y.npy")).reshape(-1) 
    classes = np.unique(Y)
    
    model = NaiveBayes(X, Y, len(classes))
    model.fit(X, Y)
    Y_pred = model.predict(X)
    print(f"Accuracy: {sum(Y_pred == Y)/X.shape[0]*100:.2f}")
