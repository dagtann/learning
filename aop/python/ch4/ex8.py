import typing


def printBinary(n: int, bitsize: int=20) -> None:
    for i in range(bitsize):
        print("1" if n & 2 ** i else "0", end="")
    return None


def main():
    N = int(input("Enter an integer:"))
    return printBinary(N)


main()
