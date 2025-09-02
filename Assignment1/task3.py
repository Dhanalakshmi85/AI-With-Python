def main():
    print("Supermarket")
    print("===========")

    # List of product prices
    prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
    total = 0

    while True:
        try:
            choice = int(input("Please select product (1-10) 0 to Quit: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if choice == 0:
            break
        elif 1 <= choice <= 10:
            price = prices[choice - 1]
            print(f"Product: {choice} Price: {price}")
            total += price
        else:
            print("Invalid product number.")

    print(f"Total: {total}")

    # Ask for payment
    while True:
        try:
            payment = int(input("Payment: "))
            if payment < total:
                print("Not enough payment. Try again.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

    change = payment - total
    print(f"Change: {change}")

# Run the program
main()
