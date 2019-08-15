import typing


def printBinary(n: int) -> None:
    if n > 1:
        printBinary(n // 2)
    print(n % 2, end = "")


def main() -> None:
    N = int(input("Input an integer:\n"))
    printBinary(N)
    return None


main()