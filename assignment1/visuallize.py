import matplotlib.pyplot as plt

def plot(f, a, b, root, method):
    x = np.linspace(a, b, 1000)
    y = f(x)
    plt.plot(x, y, label=f"$f(x) = {f.__doc__}$")
    plt.plot(root, f(root), 'o', label=f"{method} root")
    plt.legend()
    plt.show()