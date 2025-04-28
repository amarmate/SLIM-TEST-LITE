import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_synthetic1(n=500, seed=0, noise=0):
    """
    Synthetic dataset with no pieces
    """
    def f1(x): 
        return 4*x[0]**4 - 3*x[0]**3 - 5*x[0]**2 + 4*x[0] + 3
    
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, size=(n, 3))
    y_clean = f1(x.T)
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise/100) * std_dev, size=n)
    mask = [np.repeat(True, n)]
    return x, y_noisy, mask, mask


def load_synthetic2(n=500, seed=0, noise=0):
    """
    Synthetic dataset with a single threshold-based piecewise function and a slight class imbalance
    The functions are simple
    """
    def f2(x):
        if x[0] < 0.8:
            return 2 * x[0] + 1
        else:
            return -3 * x[0] + 5
        
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1
    print('Class 1 has', np.sum(x[:, 0] < 0.8), 'samples, and class 2 has', np.sum(x[:, 0] >= 0.8), 'samples') 
    y_clean = np.array([f2(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask = np.where(x[:, 0] < 0.8, True, False) 
    mask = [mask, np.logical_not(mask)]
    return x, y_noisy, mask, mask

    
def load_synthetic3(n=500, seed=0, noise=0):
    """
    Synthetic dataset with a single threshold-based piecewise function and a slight class imbalance
    The functions are harder
    """
    def f3(x): 
        if x[0] < -1: 
            return x[0]**2 
        else: 
            return x[0]**4 - 3*x[0]**3 + 2*x[0]**2 - 4
    
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1
    print('Class 1 has', np.sum(x[:, 0] < -1), 'samples, and class 2 has', np.sum(x[:, 0] >= -1), 'samples')
    y_clean = np.array([f3(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask = np.where(x[:, 0] < -1, True, False)
    mask = [mask, np.logical_not(mask)]
    return x, y_noisy, mask, mask


def load_synthetic4(n=500, seed=0, noise=0):
    """
    Synthetic dataset with a triple threshold-based piecewise function representing tax owed based on income
    Since the values are large, the dataset has to be scaled 
    """
    def f4(x): 
        if x[0] < 50000: 
            return 0.1 * x[0]
        elif x[0] < 100000:
            # return 5000 + 0.2 * (x[0] - 50000)
            return 5000 + 0.3 * (x[0] - 50000)
        elif x[0] < 150000:
            # return 15000 + 0.3 * (x[0] - 100000)
            return 20000 + 0.55 * (x[0] - 100000)
        else:
            # return 30000 + 0.5 * (x[0] - 150000)
            return 47500 + 0.8 * (x[0] - 150000)

        
    np.random.seed(seed)
    x = np.random.uniform(10000, 200000, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1
    print('Class distribution: ', np.sum(x[:, 0] < 50000), np.sum((x[:, 0] >= 50000) & (x[:, 0] < 100000)), np.sum((x[:, 0] >= 100000) & (x[:, 0] < 150000)), np.sum(x[:, 0] >= 150000))
    y_clean = np.array([f4(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)

    mask_1 = np.where(x[:, 0] < 50000, True, False)
    mask_2 = np.logical_and(x[:, 0] >= 50000, x[:, 0] < 100000)
    mask_3 = np.logical_and(x[:, 0] >= 100000, x[:, 0] < 150000)
    mask_4 = np.where(x[:, 0] >= 150000, True, False)
    mask = [mask_1, mask_2, mask_3, mask_4]
    
    # Scale X,y 
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y_noisy = scaler.fit_transform(y_noisy.reshape(-1, 1)).flatten()
    return x, y_noisy, mask, mask


def load_synthetic5(n=500, seed=0, noise=0): 
    """
    Synthetic dataset with a triple threshold-based piecewise function 
    """

    def f5(x): 
        if x[0] < 0.6: 
            return 1.4 * x[0] + 0.1
        elif x[0] < 1.2:
            return 0.9
        elif x[0] < 1.8:
            return -0.8 * x[0] + 1.85
        else:
            return -2.1 * x[0] + 4.2
    np.random.seed(seed)
    x = np.random.uniform(0, 2.5, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1

    print('Class distribution: ', np.sum(x[:, 0] < 0.6), np.sum((x[:, 0] >= 0.6) & (x[:, 0] < 1.2)), np.sum((x[:, 0] >= 1.2) & (x[:, 0] < 1.8)), np.sum(x[:, 0] >= 1.8))
    y_clean = np.array([f5(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask_1 = np.where(x[:, 0] < 0.6, True, False)

    mask_2 = np.logical_and(x[:, 0] >= 0.6, x[:, 0] < 1.2)
    mask_3 = np.logical_and(x[:, 0] >= 1.2, x[:, 0] < 1.8)
    mask_4 = np.where(x[:, 0] >= 1.8, True, False)

    mask = [mask_1, mask_2, mask_3, mask_4]
    return x, y_noisy, mask, mask 


def load_synthetic6(n=500, seed=0, noise=0): 
    """
    Synthetic dataset with a double threshold-based piecewise function 
    Harder conditions that do not depend solely on x[0]
    Easy function
    """

    def f6(x): 
        if x[1] + 2*x[2] < 4: 
            return 0.8*x[0]
        else: 
            return -0.2*x[0] + 0.1*x[1]
        
    np.random.seed(seed)
    x = np.random.uniform(0, 3, size=(n, 3))
    print('Class distribution: ', np.sum(x[:, 1] + 2*x[:, 2] < 4), np.sum(x[:, 1] + 2*x[:, 2] >= 4))
    y_clean = np.array([f6(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask_1 = np.where(x[:, 1] + 2*x[:, 2] < 4, True, False)
    mask_2 = np.logical_not(mask_1)
    mask = [mask_1, mask_2]
    return x, y_noisy, mask, mask


def load_synthetic7(n=500, seed=0, noise=0):
    """
    Synthetic dataset simulating retail sales with different regimes on weekdays versus weekends.
    - x[:,0] represents the day of the week (0 to 6, with <5 considered weekdays, >=5 as weekend).
    - x[:,1] is the promotion intensity (0 to 1).
    The functions now use quadratic adjustments instead of sine and cosine.
    """
    np.random.seed(seed)
    # Simulate day-of-week as integers from 0 to 6.
    day = np.random.randint(0, 7, size=(n, 1)).astype(float)
    # Simulate promotion intensity between 0 and 1.
    promo = np.random.uniform(0, 1, size=(n, 1))
    x = np.hstack([day, promo])
    
    def f8(x):
        # x[0]: day, x[1]: promotion intensity
        if x[0] < 5:  # Weekdays
            # Quadratic bump: (day-2.5)^2 gives a bump around the middle of the week.
            return 100 + 10 * x[0] + 20 * x[1] - 0.5 * (x[0] - 2.5)**2
        else:         # Weekends
            # Different linear and quadratic effect for weekend days.
            return 150 + 5 * x[0] + 15 * x[1] + 0.4 * (x[0] - 5)**2
    
    y_clean = np.array([f8(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)

    print('Class distribution: ', np.sum(x[:, 0] < 5), np.sum(x[:, 0] >= 5))
    
    # Create masks for weekdays and weekends
    mask_weekday = (x[:, 0] < 5)
    mask_weekend = (x[:, 0] >= 5)
    mask = [mask_weekday, mask_weekend]
    
    return x, y_noisy, mask, mask


def load_synthetic8(n=500, seed=0, noise=0):
    """
    Synthetic dataset simulating housing prices based on geographic location.
    - x contains two features: normalized latitude and longitude.
    - The condition is defined by the Euclidean distance from the city center (0.5, 0.5).
      * Urban: distance < 0.3.
      * Suburban: distance >= 0.3.
    Adjustments now use quadratic terms instead of sine/cosine.
    """
    np.random.seed(seed)
    x = np.random.uniform(0, 1, size=(n, 2))
    center = np.array([0.5, 0.5])
    
    def urban_price(x):
        # Price decays quadratically with distance from center.
        dist = np.linalg.norm(x - center)
        return 500000 * (1 - 2 * dist + 1.5 * dist**2)
    
    def suburban_price(x):
        # Price with a different quadratic adjustment beyond the threshold.
        dist = np.linalg.norm(x - center)
        # Shift the polynomial so that dist = 0.3 is the reference.
        return 300000 * (1 - (dist - 0.3)) + 50000 * (dist - 0.3)**2
    
    def f9(x):
        dist = np.linalg.norm(x - center)
        if dist < 0.3:
            return urban_price(x)
        else:
            return suburban_price(x)
    
    y_clean = np.array([f9(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    
    mask_urban = np.array([np.linalg.norm(xi - center) < 0.3 for xi in x])
    mask_suburban = np.logical_not(mask_urban)
    mask = [mask_urban, mask_suburban]

    print('Class distribution: ', np.sum(mask_urban), np.sum(mask_suburban))

    # Scale the y values to be between 0 and 1 for better interpretability.
    y_noisy = (y_noisy - np.min(y_noisy)) / (np.max(y_noisy) - np.min(y_noisy))    
    return x, y_noisy, mask, mask 


def load_synthetic9(n=500, seed=0, noise=0):
    """
    Synthetic dataset simulating an insurance risk score in a smooth, real-world fashion.
    
    Features:
      - x[:,0]: Age (range: 18 to 70)
      - x[:,1]: Driving experience in years (range: 0 to 50)
      - x[:,2]: Driving risk indicator (range: 0 to 1)
      
    The true risk score is a weighted blend of four regime-specific functions,
    with weights computed via logistic functions of age so the overall function is smooth.
    
    The four underlying regimes (which can be interpreted later) are:
      * Regime 1: Young drivers (age < ~25)
      * Regime 2: Young-to-middle-aged (age ~25 to 40)
      * Regime 3: Middle-aged drivers (age ~40 to 55)
      * Regime 4: Older drivers (age > ~55)
      
    Each regime has its own risk function (a different linear relationship) and the overall output
    is given by a smoothly weighted sum of these functions.
    """
    np.random.seed(seed)
    
    # Generate features.
    age = np.random.uniform(18, 70, n)
    experience = np.random.uniform(0, 50, n)
    driving_risk = np.random.uniform(0, 1, n)
    x = np.column_stack([age, experience, driving_risk])
    
    # Define logistic weighting functions based on age to produce smooth transitions.
    # The sharper the denominator (here 3), the more distinct the regions; we use modest slopes for smoothness.
    w1 = 1 / (1 + np.exp((age - 25) / 3))                             # High for age < ~25.
    w2 = 1 / (1 + np.exp((age - 40) / 3)) - 1 / (1 + np.exp((age - 25) / 3))  # Between ~25 and ~40.
    w3 = 1 / (1 + np.exp((age - 55) / 3)) - 1 / (1 + np.exp((age - 40) / 3))  # Between ~40 and ~55.
    w4 = 1 - 1 / (1 + np.exp((age - 55) / 3))                         # High for age > ~55.
    
    # Define regime-specific risk functions.
    # Here, coefficients are chosen for illustration so that each regime has a slightly different slope relative to the inputs.
    f1 = 0.5 * age + 0.10 * experience + 5 * driving_risk      # Regime 1: Young drivers.
    f2 = 0.6 * age + 0.05 * experience + 10 * driving_risk     # Regime 2: Transitioning to middle age.
    f3 = 0.4 * age + 0.20 * experience + 15 * driving_risk     # Regime 3: Middle-aged drivers.
    f4 = 0.7 * age + 0.15 * experience + 20 * driving_risk     # Regime 4: Older drivers.
    
    # The overall output is the smooth blend of the four regimes.
    y_clean = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4
    
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    
    # For analysis purposes, one may segment the data post hoc using simple age thresholds.
    # Here we provide four masks corresponding to approximate segmentation boundaries.
    mask1 = age < 25
    mask2 = (age >= 25) & (age < 40)
    mask3 = (age >= 40) & (age < 55)
    mask4 = age >= 55
    mask = [mask1, mask2, mask3, mask4]

    print('Class distribution: ', np.sum(mask1), np.sum(mask2), np.sum(mask3), np.sum(mask4))
    
    return x, y_noisy, mask, mask

def load_synthetic10(n=500, seed=0, noise=0):
    """
    Synthetic dataset simulating bug risk estimation for software modules.
    
    Features (all normalized to [0, 1]):
      - x[:,0]: LOC (Lines of Code)
      - x[:,1]: Complexity (Cyclomatic Complexity)
      - x[:,2]: Churn (Recent code change rate)
    
    The risk score is computed using piecewise conditions that combine AND and OR:
      Piece 1: If ((LOC > 0.7 AND Complexity > 0.7) OR (Churn > 0.8)), then:
                risk = 5*LOC + 3*Complexity + 2*Churn + 10  (High risk)
      Piece 2: Else if (LOC < 0.3 AND Churn < 0.5), then:
                risk = 1*LOC + 2*Complexity + 1.5*Churn     (Low risk)
      Piece 3: Else:
                risk = 3*LOC + 2.5*Complexity + 2*Churn + 5   (Moderate risk)
    
    Noise can be added as a percentage of the standard deviation of the clean risk scores.
    The function returns the input features, the (possibly noisy) risk score, and masks for each piece.
    """
    np.random.seed(seed)
    
    # Generate features uniformly in [0, 1] for all three attributes.
    x = np.random.uniform(0, 1, size=(n, 3))
    
    def risk_score(x):
        loc, comp, churn = x[0], x[1], x[2]
        # Piece 1: High bug risk
        if (loc > 0.7 and comp > 0.7) or (churn > 0.8):
            return 5*loc + 3*comp + 2*churn + 1
        # Piece 2: Low bug risk
        elif loc < 0.3 and churn < 0.5:
            return 1*loc + 2*comp + 1.5*churn
        # Piece 3: Moderate bug risk
        else:
            return 3*loc + 2.5*comp + 2*churn + 5
    
    # Compute the clean risk score for each sample.
    y_clean = np.array([risk_score(xi) for xi in x])
    
    # Optionally add noise based on a percentage of the output standard deviation.
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    
    # Create masks for each regime.
    # Mask for Piece 1: High risk where ((LOC > 0.7 and Complexity > 0.7) OR (Churn > 0.8))
    mask1 = np.logical_or(np.logical_and(x[:,0] > 0.7, x[:,1] > 0.7), (x[:,2] > 0.8))
    
    # Mask for Piece 2: Low risk where (LOC < 0.3 and Churn < 0.5)
    mask2 = np.logical_and(x[:,0] < 0.3, x[:,2] < 0.5)
    
    # Mask for Piece 3: The remaining cases (moderate risk)
    mask3 = ~(mask1 | mask2)

    c1 = x[:, 0] > 0.7
    c2 = x[:, 1] > 0.7
    c3 = x[:, 2] > 0.8
    c4 = x[:, 0] < 0.3
    c5 = x[:, 2] < 0.5

    
    mask = [mask1, mask2, mask3]
    condition_masks = [c1, c2, c3, c4, c5]
    print('Class distribution: ', np.sum(mask1), np.sum(mask2), np.sum(mask3))
    
    return x, y_noisy, mask, condition_masks 