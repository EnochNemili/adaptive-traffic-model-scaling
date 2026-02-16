import matplotlib.pyplot as plt

# Model sizes
models = ["Nano", "Small", "Medium", "Large"]

# Your results
low_accuracy = [0.5910, 0.7107, 0.7260, 0.7897]
high_accuracy = [0.3955, 0.5074, 0.5553, 0.5528]

# Plot Low Complexity
plt.figure()
plt.plot(models, low_accuracy)
plt.xlabel("Model Size")
plt.ylabel("Accuracy")
plt.title("Model Size vs Accuracy (Low Complexity)")
plt.ylim(0, 1)
plt.show()

# Plot High Complexity
plt.figure()
plt.plot(models, high_accuracy)
plt.xlabel("Model Size")
plt.ylabel("Accuracy")
plt.title("Model Size vs Accuracy (High Complexity)")
plt.ylim(0, 1)
plt.show()

