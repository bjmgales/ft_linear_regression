import pandas as pd
import os


def standardize(num, df: pd.DataFrame, type: str):
    return (num - df[type].mean()) / df[type].std()


def rev_standardize(num, df: pd.DataFrame, type: str):
    return num * df[type].std() + df[type].mean()


def standardize_df(df: pd.DataFrame, type: str):
    return (df[type] - df[type].mean()) / df[type].std()


def rev_standardize_df(df: pd.DataFrame, type: str):
    return df[type] * df[type].std() + df[type].mean()


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def main():
    try:
        df = pd.read_csv('./data.csv')
        if os.path.exists('./Thetas.txt') is False:
            raise FileNotFoundError('')
        file = [line.strip() for line in open('./Thetas.txt', 'r')]
        if len(file) != 2 or is_float(file[0]) is False \
                or is_float(file[1]) is False:
            raise FileNotFoundError('')

        t0 = float(file[0])
        t1 = float(file[1])

        df['km'] = standardize_df(df, 'km')

        estimate_price = t0 + (t1 * df['km'])
        estimate_price = rev_standardize(estimate_price, df, 'price')

        sum_price = df['price'].sum()
        sum_predict = estimate_price.sum()

        aggr_precision = 100 - abs(((sum_predict - sum_price)
                                    / sum_price) * 100)
        print("The model has a an aggregate prices precision of: "
              + str(aggr_precision) + "%")
        print(estimate_price)

        # Mean Absolute Value(MAE)
        mae = abs(estimate_price - df['price']).mean()
        print(f"Mean Absolute Error (MAE): {mae:.2f}")

        # Mean Squared Error (MSE)
        mse = ((df['price'] - estimate_price)**2).mean()
        print(f"Mean Squared Error (MSE): {mse:.2f}")

        # Root Mean Squared Error (RMSE)
        rmse = mse ** 0.5
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # R² Score
        ss_total = ((df['price'] - df['price'].mean())**2).sum()
        ss_residual = ((df['price'] - estimate_price)**2).sum()
        r2 = 1 - (ss_residual / ss_total)
        print(f"R² Score: {r2:.2f}")

        # Mean percentage error
        percentage_errors = abs((estimate_price - df['price'])
                                / df['price']) * 100
        accuracy = 100 - percentage_errors.mean()
        print(f"Model Accuracy: {accuracy:.2f}%")

    except FileNotFoundError:
        print("Error: you must train the model first.")
    except Exception as e:
        print(f"Error:{e}")


if __name__ == '__main__':
    main()
