import pandas as pd
import matplotlib.pyplot as plt


def standardize(df: pd.DataFrame, type: str):
    return (df[type] - df[type].mean()) / df[type].std()


def rev_standardize(df:pd.DataFrame, type: str):
    return df[type] * df[type].std() + df[type].mean()


def update_thetas(x, y, y_pred, t0, t1, learn_rate):
    tmpt0 = ((y_pred - y).sum() * 2) / len(y)
    tmpt1 = ((y_pred - y) * x).sum() * 2 / len(y)
    t0 = t0 - learn_rate * tmpt0
    t1 = t1 - learn_rate * tmpt1
    return t0, t1


def main():
    df = pd.read_csv('./data.csv')
    t0 = 0
    t1 = 0
    learn_rate = 0.1
    prev_error = float('inf')

    x = standardize(df, 'km')
    y = standardize(df, 'price')
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Data Points')
    line, = ax.plot(x, t0 + t1 * x, color='red', label='Prediction Line')
    plt.ion()
    ax.set_xlabel('Standardized km')
    ax.set_ylabel('Standardized price')
    ax.legend()

    for loop in range(10000):
        y_pred = t0 + t1 * x
        y_err = y_pred - y
        total_error = (y_err ** 2).mean()

        t0, t1 = update_thetas(x, y, y_pred, t0, t1, learn_rate)
        line.set_ydata(t0 + t1 * x)
        plt.draw()
        plt.pause(0.1)
        print(y_pred)
        if abs(prev_error - total_error) < 0.0000000000001:
            print(f"Converged after {loop + 1} loop!")
            break
        prev_error = total_error
    with open('Thetas.txt', 'w') as f:
        f.write(str(t0) + '\n' + str(t1))
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()