from math import floor


def isPrime(k):
    """ Check whether or not k is prime """
    if 0 >= k:
        return False
    for i in range(2, floor(k / 2) + 1):
        if k % i == 0:
            return False
    return True


def nextPrime(k):
    if 0 >= k:
        return None
    if not isPrime(k):
        return None
    counter = 1
    while True:
        if isPrime(k + counter):
            return k + counter
        else:
            counter += 1


def primeFactors(n):
    # n = 15
    if 1 >= n:
        return None
    checksum = n
    ans = []
    # p = 2

    if isPrime(n):
        return n
    while not isPrime(checksum):
        # assert isPrime(p), "p is not prime"
        # assert p < n, "p should always be less than n"
        for k in range(2, floor(n / 2) + 1):
            if isPrime(k) and (0 == checksum % k):
                ans.append(k)
                checksum /= k
            else: next
    return sorted(ans)

checksum = 15
checksum /= 3
checksum /= 5
checksum
def main():
    n = int(input("Enter an integer:\n"))
    return primeFactors(n)


main()
