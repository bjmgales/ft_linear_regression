import pandas as pd
import os


def standardize(num, df: pd.DataFrame, type: str):
    return (num - df[type].mean()) / df[type].std()


def rev_standardize(num, df: pd.DataFrame, type: str):
    return num * df[type].std() + df[type].mean()


def standardize_df(df: pd.DataFrame, type: str):
    return (df[type] - df[type].mean()) / df[type].std()


def rev_standardize_df(df:pd.DataFrame, type: str):
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
        val = input('Please enter a mileage: ')
        assert val.isdigit(), "invalid parameter."
        if os.path.exists('./Thetas.txt') is False:
            raise FileNotFoundError('')
        file = [line.strip() for line in open('./Thetas.txt', 'r')]
        if len(file) != 2 or is_float(file[0]) is False \
                or is_float(file[1]) is False:
            raise FileNotFoundError('')

        t0 = float(file[0])
        t1 = float(file[1])

        standardized_val = standardize(float(val), df, 'km')
        estimate_price = t0 + (t1 * standardized_val)

        estimate_price = rev_standardize(estimate_price, df, 'price')
        print(f"Estimated price for {val}km is \
{estimate_price:.2f}₳")
    except FileNotFoundError:
        print(f"Estimated price for {val}km is 0;")
    except Exception as e:
        print(f"Error:{e}")


if __name__ == '__main__':
    main()
